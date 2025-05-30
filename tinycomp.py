import re
import operator

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
VT_ANSI = {
    VT_RAW:  '\x1b[38m',
    VT_INST: '\x1b[31m',
    VT_NUM:  '\x1b[33m',
    VT_CHR:  '\x1b[32m',
    VT_PTR:  '\x1b[36m',
}


def test1():
    mem = TinyMemory(256)
    mem.set8(0, 32, VT_RAW)
    mem.set16(2, 32, VT_INST)
    mem.set16(4, 32, VT_NUM)
    mem.set16(6, 16, VT_PTR)
    mem.setstr(16, 'Hello!')
    mem.dump()


def test2():
    mem = TinyMemory(256)
    i = 0
    for part in """
        A= &0x20
        B= 10
        :A=B
        HALT
    """.split():
        val, vt = value_parse(part)
        mem.set16(i, val, vt)
        i += VT_SIZE[vt]
    proc = TinyProcessor(mem)
    proc.run()
    proc.dump()
    mem.dump()


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

def value_parse(s: str):
    if s in INSTRUCTIONS:
        return INSTRUCTIONS.index(s), VT_INST
    elif s.isdigit():
        return int(s), VT_NUM
    elif s.startswith('0x'):
        return int(s[2:], 16), VT_NUM
    elif s.startswith('&'):
        s = s[1:]
        if s.isdigit():
            return int(s), VT_PTR
        elif s.startswith('0x'):
            return int(s[2:], 16), VT_PTR
        else:
            raise Exception(f"Cannot parse: {s!r}")
    elif s.startswith('c'):
        s = s[1:]
        if s.isdigit():
            return int(s), VT_CHR
        elif s.startswith('0x'):
            return int(s[2:], 16), VT_CHR
        else:
            raise Exception(f"Cannot parse: {s!r}")
    else:
        raise Exception(f"Cannot parse: {s!r}")

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
        s = hex(value)[2:].rjust(4, '0')
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

    def __init__(self, size):
        self.size = size
        self.values = [0] * size
        self.vtypes = [VT_RAW] * size

    def get(self, i):
        vtype = self.vtypes[i]
        if VT_SIZE[vtype] == 2:
            return self.get16(i)
        else:
            return self.get8(i)

    def get8(self, i):
        return self.values[i], self.vtypes[i]

    def get16(self, i):
        return self.values[i] * 256 + self.values[i + 1], self.vtypes[i]

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

    def dump(self, i=0, *, w=16, h=16, file=None):
        max_j = i + (h - 1) * w + (w - 1)
        if i < 0 or max_j >= self.size:
            raise Exception(f"Out of range: i={i} w={w} h={h} max_j={max_j} size={self.size}")

        skip = 0
        for y in range(h):
            y_ptr = i + y * w
            print(f'{value_dump(y_ptr, VT_PTR)} | ', end='', file=file)
            for x in range(w):
                if skip > 0:
                    skip -= 1
                    #print(' ' * 2, end='', file=file)
                    continue
                j = y_ptr + x
                value, vtype = self.get(j)
                value_str = value_dump(value, vtype, just=True)
                vsize = VT_SIZE[vtype]
                if x > 0:
                    value_str = ' ' + value_str
                print(value_str, end='', file=file)
                skip = vsize - 1
            print(file=file)


class TinyProcessor:

    def __init__(self, mem: TinyMemory):
        self.mem = mem
        self.halted = False

        self.reg_values = {reg: 0 for reg in REGISTERS}
        self.reg_vtypes = {reg: VT_RAW for reg in REGISTERS}

        self.pc = 0
        self.cmp_flag = 0

    def dump(self, *, file=None):
        for reg in REGISTERS:
            value = self.reg_values[reg]
            vtype = self.reg_vtypes[reg]
            print(f'{reg}={value_dump(value, vtype)}', file=file)
        print(f'PC={value_dump(self.pc, VT_PTR)}', file=file)
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
