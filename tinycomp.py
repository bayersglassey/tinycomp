import re
import sys
import operator
from itertools import count
from argparse import ArgumentParser
from collections import defaultdict
from typing import List, Tuple, Dict, NamedTuple

MAX_I16 = 32767

def i16(i):
    """Clamp an integer to 16-bit 2's complement values

        >>> i16(2 ** 16)
        0

        >>> i16(-1)
        -1

    """
    i = i % (2 ** 16)
    return i - 2 ** 16 if i > MAX_I16 else i

# Value Types:
# Raw, instruction, number, character, pointer.
VT_RAW   = 0
VT_INST  = 1
VT_NUM   = 2
VT_CHR   = 3
VT_PTR   = 4

VT_NAME = {
    VT_RAW:  'raw',
    VT_INST: 'inst',
    VT_NUM:  'num',
    VT_CHR:  'chr',
    VT_PTR:  'ptr',
}
VTYPES_BY_NAME = {v: k for k, v in VT_NAME.items()}

VT_SIZE = {
    VT_RAW:  1,
    VT_INST: 2,
    VT_NUM:  2,
    VT_CHR:  1,
    VT_PTR:  2,
}

ANSI_RESET = '\x1b[0m'
ANSI_INVERT = '\x1b[7m'
ANSI_HIGHLIGHT = '\x1b[40m'
ANSI_NOBYTE = '\x1b[37m'
VT_ANSI = {
    VT_RAW:  '\x1b[38m',
    VT_INST: '\x1b[31m',
    VT_NUM:  '\x1b[33m',
    VT_CHR:  '\x1b[32m',
    VT_PTR:  '\x1b[36m',
}


ASM_TOKEN_REGEX = re.compile(r'\s*("(?:[^"]|\\")*"|;.*|\S+)\s*')
ASM_LABEL_REGEX = re.compile(r'[_a-zA-Z][_a-zA-Z0-9]*')


class TinyAsmParseError(Exception):
    """Error in parse_asm()"""


def vsize_sum(vtypes) -> int:
    return sum(VT_SIZE[vt] for vt in vtypes)

def vtype_repr(vtype: int) -> str:
    return VT_NAME.get(vtype, f'<unknown vtype: {vtype!r}>')

def inst_repr(value: int) -> str:
    if value < 0 or value >= len(INSTRUCTIONS):
        return None
    return INSTRUCTIONS[value]

def inst_parse(s: str) -> int:
    if s not in INSTRUCTIONS:
        return None
    return INSTRUCTIONS.index(s)

class AsmResult(NamedTuple):
    values: List[int]
    vtypes: List[int]
    labels: Dict[str, int]

def parse_asm(text: str, filename=None, *, result: AsmResult = None) -> AsmResult:
    """

        >>> result = parse_asm('''
        ...     ; I am a comment!
        ...     JMP main
        ...
        ...     main:
        ...     _start:
        ...     JNE _start
        ...     JMP _end
        ...     _end:
        ...
        ...     f:
        ...     _start:
        ...     JNE _start
        ...     JMP _end
        ...     _end:
        ...
        ...     %define x 2
        ...     %define y x
        ...     %zeros raw y
        ...
        ...     %macro values 100 -99 0xFF &0xff "hello world!"
        ...     data: values
        ... ''')
        >>> for val, vt in zip(result.values, result.vtypes):
        ...     print(value_dump(val, vt, ansi=False))
        JMP
        &0004
        JNE
        &0004
        JMP
        &000c
        JNE
        &000c
        JMP
        &0014
        00
        00
        100
        -99
        ff
        &00ff
        h
        e
        l
        l
        o
        <BLANKLINE>
        w
        o
        r
        l
        d
        !

        >>> result = parse_asm('''
        ...     %define x 2
        ...     %define y ( x + 1 )
        ...     %define z ( - y * ( 10 + 1 ) )
        ...     z
        ... ''')
        >>> for val, vt in zip(result.values, result.vtypes):
        ...     print(value_dump(val, vt, ansi=False))
        -33

    """
    if result is None:
        values = []
        vtypes = []
        labels = {}
        result = AsmResult(values, vtypes, labels)
    else:
        values, vtypes, labels = result
    definitions = {}
    macros = {}
    replacements = defaultdict(list)
    def push(value, vtype):
        values.append(value)
        vtypes.append(vtype)
    def get_ptr() -> int:
        return vsize_sum(vtypes)
    def resolve_token(token) -> str:
        while token in definitions:
            token = definitions[token]
        if token in labels:
            ptr = labels[token]
            token = f'&{hex(ptr)}'
        return token
    def do_replacements(label):
        ptr = get_ptr()
        if label in replacements:
            for i in replacements[label]:
                values[i] = ptr
            del replacements[label]
    def resolve_token_expr(tokens, token=None) -> str:
        if token is None:
            token = next(tokens)
        if token == '(':
            lhs_token = next(tokens)
            if lhs_token == '-':
                lhs_token = resolve_token_expr(tokens)
                lhs = -int(lhs_token)
            else:
                lhs_token = resolve_token_expr(tokens, lhs_token)
                lhs = int(lhs_token)
            while True:
                token = next(tokens)
                if token in BINOP_ACTIONS:
                    op_action = BINOP_ACTIONS[token]
                    rhs_token = resolve_token_expr(tokens)
                    rhs = int(rhs_token)
                    lhs = op_action(lhs, rhs)
                elif token == ')':
                    return str(lhs)
                else:
                    raise Exception(f"Expected ')' or a numerical operator")
        else:
            return resolve_token(token)
    for line_i, line in enumerate(text.splitlines()):
        token = None
        def get_tokens():
            for m in ASM_TOKEN_REGEX.finditer(line):
                token = m.group(1)
                if token in macros:
                    for token in macros[token]:
                        yield token
                else:
                    yield token
        tokens = get_tokens()
        try:
            for token in tokens:

                # Resolve any definition references, expressions, etc
                token = resolve_token_expr(tokens, token)

                if token in INSTRUCTIONS:
                    push(INSTRUCTIONS.index(token), VT_INST)
                elif token == '%define':
                    # Create a definition
                    name = next(tokens)
                    if name in definitions:
                        raise Exception(f"Redefinition of {name!r}")
                    definitions[name] = resolve_token_expr(tokens)
                elif token == '%macro':
                    # Create a macro
                    name = next(tokens)
                    if name in macros:
                        raise Exception(f"Redefinition of macro {name!r}")
                    macros[name] = list(tokens) # consume rest of line
                elif token == '%zeros':
                    # Push some number of zeros
                    vt_name = next(tokens)
                    if vt_name not in VTYPES_BY_NAME:
                        raise Exception(f"Expected one of: {', '.join(VTYPES_BY_NAME)}")
                    vtype = VTYPES_BY_NAME[vt_name]
                    number_token = resolve_token_expr(tokens)
                    if not number_token.isdigit():
                        raise Exception(f"Not a valid number: {number_token!r}")
                    number = int(number_token)
                    for i in range(number):
                        push(0, vtype)
                elif token == '%include':
                    # Include another ASM file
                    sub_filename = next(tokens)
                    sub_result = parse_asm(
                        open(sub_filename, 'r').read(),
                        sub_filename,
                        result=result,
                    )
                elif token[0] == ';':
                    # Comment
                    pass
                elif token[-1:] == ':':
                    # Defining a label
                    label = token[:-1]
                    if not ASM_LABEL_REGEX.fullmatch(label):
                        raise Exception(f"Invalid label: {label!r}")
                    if label in labels:
                        raise Exception(f"Duplicate label: {label!r}")
                    if label[0] != '_':
                        # Whenever we define a new "public label", we remove
                        # all "private labels", i.e. underscore-prefixed ones.
                        # Think of public labels as C functions, and private
                        # labels as C labels within a function.
                        ll = [l for l in labels if l.startswith('_')]
                        bad_ll = [l for l in ll if l in replacements]
                        if bad_ll:
                            raise Exception(f"Labels left undefined: {', '.join(bad_ll)}")
                        for l in ll:
                            del labels[l]
                    do_replacements(label)
                    labels[label] = get_ptr()
                elif ASM_LABEL_REGEX.fullmatch(token):
                    # Referring to a label
                    label = token
                    if label in labels:
                        # I suspect we'll never get here, since we've already
                        # said token = resolve_token(token) above...
                        # So, already-defined labels will end up being handled
                        # by the token.startswith('&') case
                        push(labels[label], VT_PTR)
                    else:
                        replacements[label].append(len(values))
                        push(0, VT_PTR)
                elif token.startswith('0x'):
                    push(int(token[2:], 16), VT_RAW)
                elif token[0] == '-' or token[0].isdigit():
                    push(int(token), VT_NUM)
                elif token.startswith('&'):
                    if token[1:3] != '0x':
                        raise Exception("Pointer literals should start with '&0x'")
                    push(int(token[3:], 16), VT_PTR)
                elif token.startswith('"'):
                    # Parse string literal
                    bslash = False
                    for c in token[1:-1]:
                        if bslash:
                            if c == 'n':
                                c = '\n'
                            elif c == '0':
                                c = '\0'
                            push(ord(c), VT_CHR)
                            bslash = False
                        elif c == '\\':
                            bslash = True
                        else:
                            push(ord(c), VT_CHR)
                else:
                    raise Exception(f"Cannot parse: {token!r}")
        except Exception as ex:
            msg = f"Parsing token {token!r} at line {line_i + 1} (remaining tokens: {list(tokens)}): {ex}"
            if filename:
                msg = f"In {filename}: {msg}"
            raise TinyAsmParseError(msg)
    if replacements:
        raise Exception(f"Labels left undefined: {', '.join(replacements)}")
    return result

def value_dump(value, vtype, *, ansi=True, just=False) -> str:
    vsize = VT_SIZE[vtype]
    if vtype == VT_RAW:
        s = hex(value)[2:].rjust(2, '0')
    elif vtype == VT_INST:
        s = inst_repr(value)
        if s is None:
            # RUH ROH
            s = '----'
    elif vtype == VT_NUM:
        s = str(value)
    elif vtype == VT_CHR:
        if value == 0:
            s = '\\0'
        elif value == 10:
            s = '\\n'
        elif value >= 127:
            s = '!!'
        else:
            s = chr(value)
            if not s.isprintable():
                s = '!!'
    elif vtype == VT_PTR:
        s = '&' + hex(value)[2:].rjust(4, '0')
    else:
        raise Exception(f"Unexpected vtype: {vtype_repr(vtype)}")
    if just:
        if vsize == 2:
            s = s.rjust(5)
        else:
            s = s.rjust(2)
    if ansi:
        ansi_code = VT_ANSI[vtype]
        s = f'{ansi_code}{s}{ANSI_RESET}'
    return s


REGISTERS = ('A', 'B', 'C', 'D')
BINOPS = '=+-*/%&|^<>?'
BINOP_ACTIONS = {
    '=': lambda x, y: y,
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.floordiv,
    '%': operator.mod,
    '&': operator.and_,
    '|': operator.or_,
    '&': operator.xor,
    '<': operator.lshift,
    '>': operator.rshift,
    '?': operator.eq,
}
LARGS = ('PC', 'IO', '.', ':') + tuple(
    f'{prefix}{register}'
    for register in REGISTERS
    for prefix in ('', '.', ':'))
RARGS = LARGS + ('', '0', '1', '2')
CHAR_ARGS = ('IO',) + tuple(f'.{reg}' for reg in REGISTERS)
BINOP_REGEX = re.compile(
    f"({'|'.join(re.escape(arg) for arg in LARGS)})"
    + f"({'|'.join(re.escape(op) for op in BINOPS)})"
    + f"({'|'.join(re.escape(arg) for arg in RARGS)})")
JUMP_INSTRUCTIONS = ('JMP', 'JE', 'JNE', 'JL', 'JLE', 'JG', 'JGE')
OTHER_INSTRUCTIONS = ('NOOP', 'HALT', 'BKPT')
INSTRUCTIONS = (
    OTHER_INSTRUCTIONS
    + JUMP_INSTRUCTIONS
    + tuple(
        f'{larg}{op}{rarg}'
        for op in BINOPS
        for larg in LARGS
        for rarg in RARGS)
)


JUMP_INSTRUCTIONS_ACCEPTED_CMP = {
    'JMP': (-1, 0, 1),
    'JE': (0,),
    'JNE': (-1, 1),
    'JL': (-1,),
    'JLE': (0, -1),
    'JG': (1,),
    'JGE': (0, 1),
}


class TinyMemory:
    """

        >>> mem = TinyMemory(16 * 4)
        >>> mem.set8(0, 32, VT_RAW)
        >>> mem.set16(2, INSTRUCTIONS.index('IO=B'), VT_INST)
        >>> mem.set16(4, 32, VT_NUM)
        >>> mem.set16(6, 16, VT_PTR)
        >>> mem.setstr(16, 'Hello!')
        >>> mem.dump(ansi=False, bars=True)
             +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
        0000 |20|00| IO=B|   32|&0010|00|00|00|00|00|00|00|00|
        0010 | H| e| l| l| o| !|00|00|00|00|00|00|00|00|00|00|
        0020 |00|00|00|00|00|00|00|00|00|00|00|00|00|00|00|00|
        0030 |00|00|00|00|00|00|00|00|00|00|00|00|00|00|00|00|
             +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
        >>> mem.dump(ansi=False, bars=True, raw=True)
             +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
        0000 |20|00|00|25|00|20|00|10|00|00|00|00|00|00|00|00|
        0010 |48|65|6c|6c|6f|21|00|00|00|00|00|00|00|00|00|00|
        0020 |00|00|00|00|00|00|00|00|00|00|00|00|00|00|00|00|
        0030 |00|00|00|00|00|00|00|00|00|00|00|00|00|00|00|00|
             +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+

        >>> from io import BytesIO
        >>> file = BytesIO()
        >>> mem.save(file)
        >>> file.seek(0)
        0
        >>> mem2 = TinyMemory.load(file)
        >>> mem == mem2
        True

    """

    def __init__(self, size):
        self.size = size
        self.values = [0] * size
        self.vtypes = [VT_RAW] * size
        self.changes = set()
        self.accesses = set()

    def __eq__(self, other):
        return self.values == other.values and self.vtypes == other.vtypes

    def reset_changes_and_accesses(self):
        self.changes.clear()
        self.accesses.clear()

    def write_values(self, i, values, vtypes):
        size = len(values)
        if size != len(vtypes):
            raise Exception("Length of values and vtypes must match")
        for value, vtype in zip(values, vtypes):
            self.set(i, value, vtype)
            i += VT_SIZE[vtype]
        self.reset_changes_and_accesses()

    def write_asm(self, i, result: AsmResult):
        return self.write_values(i, result.values, result.vtypes)

    def save(self, file):
        if isinstance(file, str):
            file = open(file, 'wb')
        parts = []
        for value, vtype in zip(self.values, self.vtypes):
            word = vtype * 256 + value
            parts.append(word.to_bytes(2, 'big'))
        data = b''.join(parts)
        file.write(data)

    @staticmethod
    def load_asm(file_or_asm, size=None):
        if isinstance(file_or_asm, AsmResult):
            asm = file_or_asm
        elif file_or_asm == '-':
            filename = '<STDIN>'
            file = sys.stdin
            asm = parse_asm(file.read(), filename)
        elif isinstance(file_or_asm, str):
            filename = file_or_asm
            file = open(filename, 'r')
            asm = parse_asm(file.read(), filename)
        else:
            file = file_or_asm
            asm = parse_asm(file.read())
        if size is None:
            size = vsize_sum(asm.vtypes)
        self = TinyMemory(size)
        self.write_asm(0, asm)
        return self

    @staticmethod
    def load(file, size=None):
        if file == '-':
            file = sys.stdin
        elif isinstance(file, str):
            file = open(file, 'rb')
        data = file.read()
        if size is None:
            size = len(data) // 2
        self = TinyMemory(size)
        for i in range(size):
            word = int.from_bytes(data[i*2:i*2+2], 'big')
            vtype = word // 256
            value = word % 256
            self.values[i] = value
            self.vtypes[i] = vtype
        return self

    def get(self, i, **kwargs):
        vtype = self.vtypes[i]
        if VT_SIZE[vtype] == 2:
            return self.get16(i, **kwargs)
        else:
            return self.get8(i, **kwargs)

    def get8(self, i, quiet=False):
        if not quiet:
            self.accesses.add(i)
        return self.values[i], self.vtypes[i]

    def get16(self, i, quiet=False):
        if not quiet:
            self.accesses.add(i)
            self.accesses.add(i + 1)
        vtype = self.vtypes[i]
        value = self.values[i] * 256 + self.values[i + 1]
        if vtype == VT_NUM:
            value = i16(value)
        return value, vtype

    def set(self, i, value, vtype):
        if VT_SIZE[vtype] == 2:
            self.set16(i, value, vtype)
        else:
            self.set8(i, value, vtype)

    def set8(self, i, value, vtype):
        if VT_SIZE[vtype] != 1:
            # Make sure we use an 8-bit vtype
            vtype = VT_RAW
        self.values[i] = value % 256
        self.vtypes[i] = vtype
        self.changes.add(i)

    def set16(self, i, value, vtype):
        if VT_SIZE[vtype] != 2:
            # Make sure we use a 16-bit vtype?..
            # (or, since given vtype is 8-bit, I guess we could assume
            # that value % 256 == value, and just keep vtype as-is?..)
            vtype = VT_NUM
        value = value % (2 ** 16)
        self.values[i] = value // 256
        self.values[i + 1] = value % 256
        self.vtypes[i] = vtype
        self.vtypes[i + 1] = VT_RAW
        self.changes.add(i)
        self.changes.add(i + 1)

    def setstr(self, i, s):
        for c in s:
            value = ord(c)
            self.set8(i, value, VT_CHR)
            i += 1

    def dump(self, *,
            w=16,
            h=None,
            ansi=True,
            file=None,
            bars=False,
            raw=False,
            rows=None,
            highlight_accesses=True,
            extra_highlights=(),
            ):
        space = '|' if bars else ' '
        hbar = '     +' + '--+' * w

        def print_value_str(j, value_str):
            if ansi:
                if highlight_accesses and (j in self.accesses or j in self.changes):
                    value_str = f'{ANSI_HIGHLIGHT}{value_str}'
                if j in extra_highlights:
                    value_str = f'{ANSI_INVERT}{value_str}'
            print(value_str, end='', file=file)

        skip = 0
        print(hbar, file=file)
        printed_ellipses = False
        for y in count() if h is None else range(h):
            y_ptr = y * w
            if y_ptr >= self.size:
                break
            if rows is not None and y not in rows:
                if not printed_ellipses:
                    print(' ...', file=file)
                    printed_ellipses = True
                continue
            else:
                printed_ellipses = False
            y_ptr_s = value_dump(y_ptr, VT_PTR, ansi=ansi).replace('&', '')
            print(f'{y_ptr_s} |', end='', file=file)
            for x in range(w):
                j = y_ptr + x
                if skip > 0:
                    if x == 0:
                        value_str = '<<'
                        if ansi:
                            value_str = f'{ANSI_NOBYTE}{value_str}{ANSI_RESET}'
                        print_value_str(j, value_str)
                    skip -= 1
                    continue
                if x > 0:
                    print(space, end='', file=file)
                if j >= self.size:
                    value_str = '##'
                    if ansi:
                        value_str = f'{ANSI_NOBYTE}{value_str}{ANSI_RESET}'
                    print_value_str(j, value_str)
                    continue
                if raw:
                    value = self.values[j]
                    vtype = VT_RAW
                else:
                    value, vtype = self.get(j, quiet=True)
                value_str = value_dump(value, vtype, ansi=ansi, just=True)
                print_value_str(j, value_str)
                vsize = VT_SIZE[vtype]
                skip = vsize - 1
            print('|', file=file)
        print(hbar, file=file)


class TinyTerminal:

    def __init__(self, infile=None, outfile=None):
        self.infile = infile or sys.stdin
        self.outfile = outfile or sys.stdout
        self._should_flush = callable(getattr(self.outfile, 'flush', None))

    def getch(self) -> int:
        b = self.infile.read(1).encode()
        return int.from_bytes(b, 'big')

    def putch(self, value):
        self.outfile.write(chr(value))
        if self._should_flush:
            self.outfile.flush()


class TinyProcessor:
    """

        >>> result = parse_asm('''
        ...     A= &0x20
        ...     B= 10
        ...     :A=B
        ...     HALT
        ... ''')

        >>> mem = TinyMemory(16 * 4)
        >>> mem.write_asm(0, result)
        >>> proc = TinyProcessor(mem)
        >>> proc.run()
        >>> proc.dump_registers(ansi=False)
        A=&0020
        B=10
        C=00
        D=00
        PC=&000a
        CMP=0
        >>> proc.dump_mem(ansi=False)
             +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
        0000 |   A= &0020    B=    10  :A=B  HALT 00 00 00 00|
        0010 |00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00|
        0020 |   10 00 00 00 00 00 00 00 00 00 00 00 00 00 00|
        0030 |00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00|
             +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+

    """

    def __init__(self, mem: TinyMemory, *, term: TinyTerminal = None):
        self.mem = mem
        self.term = term or TinyTerminal()

        self.halted = False
        self.stepping = False
        self._dump = None
        self.breakpoints = set()

        self.reg_values = {reg: 0 for reg in REGISTERS}
        self.reg_vtypes = {reg: VT_RAW for reg in REGISTERS}
        self.reg_changes = {reg: False for reg in REGISTERS}

        self.pc = 0
        self.cmp_flag = 0

    def add_breakpoint(self, i):
        self.breakpoints.add(i)

    def reset_changes_and_accesses(self):
        self.mem.reset_changes_and_accesses()
        for reg in REGISTERS:
            self.reg_changes[reg] = False

    def dump_mem(self, **kwargs):
        self.mem.dump(extra_highlights=(self.pc,), **kwargs)

    def dump_registers(self, *, ansi=True, file=None):
        for reg in REGISTERS:
            value = self.reg_values[reg]
            vtype = self.reg_vtypes[reg]
            value_str = value_dump(value, vtype, ansi=ansi)
            if ansi and self.reg_changes[reg]:
                value_str = f'{ANSI_HIGHLIGHT}{value_str}'
            print(f'{reg}={value_str}', file=file)
        print(f'PC={value_dump(self.pc, VT_PTR, ansi=ansi)}', file=file)
        print(f'CMP={self.cmp_flag}', file=file)

    def step(self):
        self.reset_changes_and_accesses()
        pc_value, pc_vtype = self.mem.get16(self.pc)
        if pc_value >= len(INSTRUCTIONS):
            self.halted = True
            return

        inst = INSTRUCTIONS[pc_value]
        if inst == 'HALT':
            self.halted = True
        elif inst == 'NOOP':
            self.pc += 2
        elif inst == 'BKPT':
            self.stepping = True
            self.pc += 2
        elif inst in JUMP_INSTRUCTIONS:
            matched = self.cmp_flag in JUMP_INSTRUCTIONS_ACCEPTED_CMP[inst]
            ptr, ptr_vtype = self.mem.get16(self.pc + 2)
            if matched:
                self.pc = ptr
            else:
                self.pc += 4
        else:
            match = BINOP_REGEX.fullmatch(inst)
            if not match:
                # This should never happen, unless we add an instruction and
                # forget to implement it
                raise Exception(f"Unrecognized instruction: {pc_value!r}")
            larg, op, rarg = match.groups()
            old_pc = self.pc
            self.pc += 2
            arg_size = 1 if larg in CHAR_ARGS else 2
            def get_arg_param(arg):
                if arg in ('', '.', ':'):
                    if arg_size == 2:
                        ret = self.mem.get16(self.pc)
                    else:
                        ret = self.mem.get8(self.pc)
                    self.pc += arg_size
                    return ret
                else:
                    return None, None
            lparam, lpvtype = get_arg_param(larg)
            rparam, rpvtype = get_arg_param(rarg)
            if op == '=':
                # Avoid getting the lvalue's old value, which in particular
                # avoids calling self.term.getch() when larg is 'IO'.
                lvalue = 0
                ltype = VT_RAW
            else:
                lvalue, lvtype = self._get_arg(larg, lparam, lpvtype, old_pc)
            rvalue, rvtype = self._get_arg(rarg, rparam, rpvtype, old_pc)
            if op == '?':
                if lvalue < rvalue:
                    self.cmp_flag = -1
                elif lvalue > rvalue:
                    self.cmp_flag = 1
                else:
                    self.cmp_flag = 0
            else:
                value = BINOP_ACTIONS[op](lvalue, rvalue)
                vtype = rvtype if op == '=' else lvtype
                self._set_arg(larg, lparam, lpvtype, value, vtype)

    def set_dump(self, dump=None):
        self._dump = dump

    def dump(self):
        if self._dump is not None:
            self._dump()
        else:
            self.dump_mem()
            self.dump_registers()

    def run(self):
        prev_cmd = ''
        while True:
            if self.pc in self.breakpoints:
                self.stepping = True
            if self.stepping:
                self.dump()
                while self.stepping:
                    cmd = input("Debug mode. Enter a command ('h' to see all commands): ")
                    if cmd:
                        prev_cmd = cmd
                    else:
                        cmd = prev_cmd
                    for c in cmd:
                        if c == 'd':
                            self.dump()
                        elif c == 'c':
                            self.stepping = False
                            prev_cmd = ''
                        elif c == 'h':
                            print("Debug mode commands:")
                            print(" h: help (show this message)")
                            print(" d: dump/display memory & registers")
                            print(" n: execute next instruction")
                            print(" c: continue execution (exit debug mode)")
                        elif c in ('n', ''):
                            self.step()
            else:
                if self.halted:
                    break
                self.step()

    def _get_arg(self, arg, param, pvtype, old_pc):
        if arg == '':
            return param, pvtype
        elif arg in '012':
            return int(arg), VT_NUM
        elif arg in 'ABCD':
            return self.reg_values[arg], self.reg_vtypes[arg]
        elif arg[0] in '.:':
            if len(arg) == 2:
                reg = arg[1]
                ptr = self.reg_values[reg]
            else:
                ptr = param
            if arg[0] == '.':
                return self.mem.get8(ptr)
            else:
                return self.mem.get16(ptr)
        elif arg == 'PC':
            return old_pc, VT_PTR
        elif arg == 'IO':
            return self.term.getch(), VT_CHR
        else:
            raise Exception(f"Unrecognized arg: {arg}")

    def _set_reg(self, reg, value, vtype):
        self.reg_values[reg] = value
        self.reg_vtypes[reg] = vtype
        self.reg_changes[reg] = True

    def _set_arg(self, arg, param, pvtype, value, vtype):
        if arg in 'ABCD':
            self._set_reg(arg, value, vtype)
        elif arg[0] in '.:':
            if len(arg) == 2:
                reg = arg[1]
                ptr = self.reg_values[reg]
            else:
                ptr = param
            if arg[0] == '.':
                self.mem.set8(ptr, value, vtype)
            else:
                self.mem.set16(ptr, value, vtype)
        elif arg == 'PC':
            self.pc = value
        elif arg == 'IO':
            self.term.putch(value)
        else:
            raise Exception(f"Unrecognized arg: {arg}")


def parse_cli_args():
    parser = ArgumentParser()
    parser.add_argument('--file', '-f', required=True)
    parser.add_argument('--size', default=None, type=int)
    parser.add_argument('--mem', '-m', default=False, action='store_true')
    parser.add_argument('--run', '-r', default=False, action='store_true')
    parser.add_argument('--ansi', default=True, action='store_false')
    parser.add_argument('--bars', default=False, action='store_true')
    parser.add_argument('--labels', default=False, action='store_true')
    parser.add_argument('--raw', default=False, action='store_true')
    parser.add_argument('--quiet', '-q', default=False, action='store_true')
    parser.add_argument('-b', '--bkpt', nargs='+',
        help="List of breakpoints, either decimal numbers, hex numbers with '0x' prefix, or ASM label names")
    parser.add_argument('-H', '--highlights', nargs='+',
        help="List of ASM label names; those sections of memory are kept permanently highlighted")
    return parser.parse_args()


def main(args=None):
    if args is None:
        args = parse_cli_args()
    if args.mem:
        asm = None
        mem = TinyMemory.load(args.file, args.size)
        proc = None
    else:
        if args.file == '-':
            filename = '<STDIN>'
            file = sys.stdin
        else:
            filename = args.file
            file = open(filename, 'r')
        asm = parse_asm(file.read(), filename)
        file.close()
        mem = TinyMemory.load_asm(asm, args.size)
        proc = TinyProcessor(mem)
    for bkpt in args.bkpt or ():
        try:
            if bkpt.isdigit():
                bkpt = int(bkpt)
            elif bkpt.startswith('0x'):
                bkpt = int(bkpt[2:], 16)
            else:
                if bkpt not in asm.labels:
                    raise Exception(f"Label {bkpt!r} not found. Available are: {', '.join(asm.labels)}")
                bkpt = asm.labels[bkpt]
            proc.add_breakpoint(bkpt)
        except Exception as ex:
            raise Exception(f"Couldn't set breakpoint {bkpt!r}: {ex}")

    highlight_rows = set()
    for hlight in args.highlights or ():
        try:
            if hlight not in asm.labels:
                raise Exception(f"Label {hlight!r} not found. Available are: {', '.join(asm.labels)}")
            ptr = asm.labels[hlight]
            label_i = list(asm.labels).index(hlight)
            if label_i < len(asm.labels) - 1:
                end = list(asm.labels.values())[label_i + 1]
            else:
                end = mem.size
            for i in range(ptr, end):
                highlight_rows.add(i // 16)
        except Exception as ex:
            raise Exception(f"Couldn't set highlight {hlight!r}: {ex}")

    mem_dump_kwargs = dict(ansi=args.ansi, bars=args.bars, raw=args.raw)
    def dump():
        if proc is not None:
            if proc.stepping:
                dump_rows = highlight_rows.copy()
                def _add_row(y, buffer=1):
                    for i in range(y - buffer, y + buffer + 1):
                        dump_rows.add(i)
                _add_row(proc.pc // 16)
                for i in proc.mem.accesses:
                    _add_row(i // 16)
            else:
                dump_rows = None
            proc.dump_mem(rows=dump_rows, **mem_dump_kwargs)
            proc.dump_registers(ansi=args.ansi)
            if args.labels and asm.labels:
                print("labels:")
                for name, ptr in asm.labels.items():
                    print(f"  {name}={value_dump(ptr, VT_PTR, ansi=args.ansi)}")
        else:
            mem.dump(**mem_dump_kwargs)
    if proc is not None:
        proc.set_dump(dump)

    if args.run:
        proc.run()

    if not args.quiet:
        dump()


if __name__ == '__main__':
    main()
