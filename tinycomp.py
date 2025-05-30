import math

# Value Types:
# Raw, instruction, number, character, pointer.
VT_RAW   = 0
VT_INST  = 1
VT_NUM   = 2
VT_CHR   = 3
VT_PTR   = 4

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


def instr_repr(value):
    return 'INSTR' # TODO


class TinyMemory:

    def __init__(self, size):
        self.size = size
        self.values = [0] * size
        self.vtypes = [VT_RAW] * size

    def get8(self, i):
        return self.values[i]

    def get16(self, i):
        return self.values[i] * 256 + self.values[i + 1]

    def set8(self, i, value, vtype):
        self.values[i] = value % 256
        self.vtypes[i] = vtype

    def set16(self, i, value, vtype):
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

        ptr_ansi = VT_ANSI[VT_PTR]
        ptr_width = math.ceil(math.log(self.size, 16))

        skip = 0
        for y in range(h):
            y_ptr = hex(i + y * w)[2:].rjust(ptr_width, '0')
            print(f'{ptr_ansi}{y_ptr}{ANSI_RESET} | ', end='', file=file)
            for x in range(w):
                if skip > 0:
                    skip -= 1
                    #print(' ' * 2, end='', file=file)
                    continue
                elif x > 0:
                    print(' ', end='', file=file)
                j = i + y * w + x
                vtype = self.vtypes[j]
                vsize = VT_SIZE[vtype]
                v_ansi = VT_ANSI[vtype]
                if vsize == 2:
                    value = self.get16(j)
                    if vtype == VT_INST:
                        value = instr_repr(value)
                    elif vtype == VT_NUM:
                        value = str(value).rjust(5)
                    elif vtype == VT_PTR:
                        value = hex(value)[2:].rjust(5, '0')
                    else:
                        raise Exception(f"Unexpected vtype: {vtype!r}")
                    print(f'{v_ansi}{value}{ANSI_RESET}', end='', file=file)
                    skip = 1
                elif vsize == 1:
                    value = self.get8(j)
                    if vtype == VT_RAW:
                        value = hex(value)[2:].rjust(2, '0')
                    elif vtype == VT_CHR:
                        if value < 127:
                            value = chr(value)
                            if not value.isprintable():
                                value = '.'
                        else:
                            value = '.'
                        value = ' ' + value
                    else:
                        raise Exception(f"Unexpected vtype: {vtype!r}")
                    print(f'{v_ansi}{value}{ANSI_RESET}', end='', file=file)
                else:
                    raise Exception(f"Unexpected vsize: {vsize!r}")
            print(file=file)
