import re
import operator
from collections import defaultdict
from typing import List, Tuple

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

VT_SIZE = {
    VT_RAW:  1,
    VT_INST: 2,
    VT_NUM:  2,
    VT_CHR:  1,
    VT_PTR:  2,
}

ANSI_RESET = '\x1b[0m'
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

def parse_asm(text: str, filename=None) -> Tuple[List[int], List[int]]:
    """

        >>> values, vtypes = parse_asm('''
        ...     ; I am a comment!
        ...     %define x 100
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
        ...     data: x -99 0xFF &0xff "hello world!"
        ... ''')
        >>> for val, vt in zip(values, vtypes):
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

    """
    values = []
    vtypes = []
    defines = {}
    labels = {}
    replacements = defaultdict(list)
    def add(value, vtype):
        values.append(value)
        vtypes.append(vtype)
    def get_ptr():
        return sum(VT_SIZE[vt] for vt in vtypes)
    def do_replacements(label):
        ptr = get_ptr()
        if label in replacements:
            for i in replacements[label]:
                values[i] = ptr
            del replacements[label]
    for line_i, line in enumerate(text.splitlines()):
        token = None
        try:
            tokens = (m.group(1) for m in ASM_TOKEN_REGEX.finditer(line))
            for token in tokens:

                # Reference to a definition
                if token in defines:
                    token = defines[token]

                if token in INSTRUCTIONS:
                    add(INSTRUCTIONS.index(token), VT_INST)
                elif token == '%define':
                    name = next(tokens)
                    defines[name] = next(tokens)
                elif token == '%include':
                    sub_filename = next(tokens)
                    sub_values, sub_vtypes = parse_asm(
                        open(sub_filename, 'r').read(),
                        sub_filename)
                    values += sub_values
                    vtypes += sub_vtypes
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
                        add(labels[label], VT_PTR)
                    else:
                        replacements[label].append(len(values))
                        add(0, VT_PTR)
                elif token.startswith('0x'):
                    add(int(token[2:], 16), VT_RAW)
                elif token[0] == '-' or token[0].isdigit():
                    add(int(token), VT_NUM)
                elif token.startswith('&'):
                    if token[1:3] != '0x':
                        raise Exception("Pointer literals should start with '&0x'")
                    add(int(token[3:], 16), VT_PTR)
                elif token.startswith('"'):
                    bslash = False
                    for c in token[1:-1]:
                        if bslash:
                            if c == 'n':
                                c = '\n'
                            add(ord(c), VT_CHR)
                            bslash = False
                        elif c == '\\':
                            bslash = True
                        else:
                            add(ord(c), VT_CHR)
                else:
                    raise Exception(f"Cannot parse: {token!r}")
        except Exception as ex:
            msg = f"Parsing token {token!r} at line {line_i + 1}: {ex}"
            if filename:
                msg = f"In {filename}: {msg}"
            raise TinyAsmParseError(msg)
    if replacements:
        raise Exception(f"Labels left undefined: {', '.join(replacements)}")
    return values, vtypes

def value_dump(value, vtype, *, ansi=True, just=False) -> str:
    vsize = VT_SIZE[vtype]
    if vtype == VT_RAW:
        s = hex(value)[2:].rjust(2, '0')
    elif vtype == VT_INST:
        s = inst_repr(value)
    elif vtype == VT_NUM:
        s = str(value)
    elif vtype == VT_CHR:
        if value < 127:
            s = chr(value)
            if not s.isprintable():
                s = '.'
        else:
            s = '.'
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
LARGS = ('PC', 'IO') + tuple(
    f'{prefix}{register}'
    for register in REGISTERS
    for prefix in ('', '.', ':'))
RARGS = LARGS + ('', '0', '1', '2')
BINOP_REGEX = re.compile(
    f"({'|'.join(re.escape(arg) for arg in LARGS)})"
    + f"({'|'.join(re.escape(op) for op in BINOPS)})"
    + f"({'|'.join(re.escape(arg) for arg in RARGS)})")
JUMP_INSTRUCTIONS = ('JMP', 'JE', 'JNE', 'JL', 'JLE', 'JG', 'JGE')
OTHER_INSTRUCTIONS = ('NOOP', 'HALT')
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
        >>> mem.set16(2, 32, VT_INST) # IO=B
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
        0000 |20|00|00|20|00|20|00|10|00|00|00|00|00|00|00|00|
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

    def __eq__(self, other):
        return self.values == other.values and self.vtypes == other.vtypes

    def write_values(self, i, values, vtypes):
        size = len(values)
        if size != len(vtypes):
            raise Exception("Length of values and vtypes must match")
        for value, vtype in zip(values, vtypes):
            self.set(i, value, vtype)
            i += VT_SIZE[vtype]

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
    def load_asm(file, size=None):
        if isinstance(file, str):
            filename = file
            file = open(file, 'r')
        else:
            filename = None
        values, vtypes = parse_asm(file.read(), filename)
        if size is None:
            size = sum(VT_SIZE[vt] for vt in vtypes)
        self = TinyMemory(size)
        self.write_values(0, values, vtypes)
        return self

    @staticmethod
    def load(file):
        if isinstance(file, str):
            file = open(file, 'rb')
        data = file.read()
        size = len(data) // 2
        self = TinyMemory(size)
        for i in range(size):
            word = int.from_bytes(data[i*2:i*2+2], 'big')
            vtype = word // 256
            value = word % 256
            self.values[i] = value
            self.vtypes[i] = vtype
        return self

    def get(self, i):
        vtype = self.vtypes[i]
        if VT_SIZE[vtype] == 2:
            return self.get16(i)
        else:
            return self.get8(i)

    def get8(self, i):
        return self.values[i], self.vtypes[i]

    def get16(self, i):
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

    def setstr(self, i, s):
        for c in s:
            value = ord(c)
            self.set8(i, value, VT_CHR)
            i += 1

    def dump(self, i=0, *, w=16, h=16, ansi=True, file=None, bars=False, raw=False):
        space = '|' if bars else ' '
        hbar = '     +' + '--+' * w
        skip = 0
        print(hbar, file=file)
        for y in range(h):
            y_ptr = i + y * w
            if y_ptr >= self.size:
                break
            y_ptr_s = value_dump(y_ptr, VT_PTR, ansi=ansi).replace('&', '')
            print(f'{y_ptr_s} |', end='', file=file)
            for x in range(w):
                if skip > 0:
                    skip -= 1
                    continue
                if x > 0:
                    print(space, end='', file=file)
                j = y_ptr + x
                if j >= self.size:
                    value_str = '##'
                    if ansi:
                        value_str = f'{ANSI_NOBYTE}{value_str}{ANSI_RESET}'
                    print(value_str, end='', file=file)
                    continue
                if raw:
                    value = self.values[j]
                    vtype = VT_RAW
                else:
                    value, vtype = self.get(j)
                value_str = value_dump(value, vtype, ansi=ansi, just=True)
                print(value_str, end='', file=file)
                vsize = VT_SIZE[vtype]
                skip = vsize - 1
            print('|', file=file)
        print(hbar, file=file)


class TinyProcessor:
    """

        >>> values, vtypes = parse_asm('''
        ...     A= &0x20
        ...     B= 10
        ...     :A=B
        ...     HALT
        ... ''')

        >>> mem = TinyMemory(16 * 4)
        >>> mem.write_values(0, values, vtypes)
        >>> proc = TinyProcessor(mem)
        >>> proc.run()
        >>> proc.dump(ansi=False)
        A=&0020
        B=10
        C=00
        D=00
        PC=&000a
        CMP=0
        >>> mem.dump(ansi=False)
             +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
        0000 |   A= &0020    B=    10  :A=B  HALT 00 00 00 00|
        0010 |00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00|
        0020 |   10 00 00 00 00 00 00 00 00 00 00 00 00 00 00|
        0030 |00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00|
             +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+

    """

    def __init__(self, mem: TinyMemory):
        self.mem = mem
        self.halted = False

        self.reg_values = {reg: 0 for reg in REGISTERS}
        self.reg_vtypes = {reg: VT_RAW for reg in REGISTERS}

        self.pc = 0
        self.cmp_flag = 0

    def dump(self, *, ansi=True, file=None):
        for reg in REGISTERS:
            value = self.reg_values[reg]
            vtype = self.reg_vtypes[reg]
            print(f'{reg}={value_dump(value, vtype, ansi=ansi)}', file=file)
        print(f'PC={value_dump(self.pc, VT_PTR, ansi=ansi)}', file=file)
        print(f'CMP={self.cmp_flag}', file=file)

    def step(self):
        pc_value, pc_vtype = self.mem.get16(self.pc)
        if pc_value >= len(INSTRUCTIONS):
            self.halted = True
            return
        inst = INSTRUCTIONS[pc_value]
        if inst == 'HALT':
            self.halted = True
            return
        elif inst == 'NOOP':
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
            lvalue, lvtype = self._getarg(larg)
            rvalue, rvtype = self._getarg(rarg)
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
                self._setarg(larg, value, vtype)
            if rarg == '':
                self.pc += 4
            else:
                self.pc += 2

    def run(self):
        while not self.halted:
            self.step()

    def _getarg(self, arg):
        if arg == '':
            return self.mem.get16(self.pc + 2)
        elif arg in '012':
            return int(arg), VT_NUM
        elif arg in 'ABCD':
            return self.reg_values[arg], self.reg_vtypes[arg]
        elif arg[0] in '.:':
            reg = arg[1]
            ptr = self.reg_values[reg]
            if arg[0] == '.':
                return self.mem.get8(ptr)
            else:
                return self.mem.get16(ptr)
        elif arg == 'PC':
            return self.pc, VT_PTR
        elif arg == 'IO':
            # TODO (self.term or something)
            raise Exception("I/O not implemented yet")
        else:
            raise Exception(f"Unrecognized arg: {arg}")

    def _setarg(self, arg, value, vtype):
        if arg in 'ABCD':
            self.reg_values[arg] = value
            self.reg_vtypes[arg] = vtype
        elif arg[0] in '.:':
            reg = arg[1]
            ptr = self.reg_values[reg]
            if arg[0] == '.':
                self.mem.set8(ptr, value, vtype)
            else:
                self.mem.set16(ptr, value, vtype)
        elif arg == 'PC':
            raise Exception("Can't set PC directly... use JMP instead")
        elif arg == 'IO':
            # TODO (self.term or something)
            raise Exception("I/O not implemented yet")
        else:
            raise Exception(f"Unrecognized arg: {arg}")
