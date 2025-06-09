import re
import sys
from argparse import ArgumentParser
from typing import List, Tuple, Dict, NamedTuple, Optional, Any
from collections import deque

from tinycomp import *


UNARY_OP = 'unary'
BINARY_OP = 'binary'

class Operator(NamedTuple):
    name: str
    token: Optional[str]
    arity: str # UNARY_OP, BINARY_OP, etc
    prec: int # precedence
    assoc_rtol: bool = False # associates right-to-left?

def prec_check(prev_op: Optional[Operator], op: Operator) -> bool:
    """Whether op takes associative precedence over prev_op.
    E.g. in 'x + y + z', prec_check returns True if this should be interpreted
    as 'x + (y + z)', or False if it should be interpreted as '(x + y) + z'.

        >>> def test(lop, rop):
        ...     def get_op(op):
        ...         if op == '(':
        ...             return OPS_BY_NAME['call']
        ...         elif op == '[':
        ...             return OPS_BY_NAME['index']
        ...         elif op in OPS_BY_NAME:
        ...             return OPS_BY_NAME[op]
        ...         else:
        ...             return BINOPS_BY_TOKEN[op]
        ...     return prec_check(get_op(lop), get_op(rop))

        E.g. 'x + y + z' -> '(x + y) + z'
        >>> test('+', '+')
        False

        E.g. 'x = y = z' -> 'x = (y = z)'
        >>> test('=', '=')
        True

        E.g. 'x * y + z' -> '(x * y) + z'
        >>> test('*', '+')
        False

        E.g. 'x + y * z' -> 'x + (y * z)'
        >>> test('+', '*')
        True

        E.g. '*x[3]' -> '*(x[3])'
        >>> test('deref', 'index')
        True

    """
    if prev_op is None:
        return True
    elif op.prec == prev_op.prec:
        return op.assoc_rtol
    else:
        return op.prec < prev_op.prec


# The basic tinc types, and their corresponding tinycomp "value types".
# Pointers aren't listed here, because from tinc's perspective, they're
# always pointer-to-something.
BASIC_TYPES = {
    'void': None,
    'char': VT_CHR,
    'byte': VT_RAW,
    'int': VT_NUM,
    'inst': VT_INST,
}

OPS = (
    # NOTE: ops whose token starts with the token of another op should
    # come *before* that other op in this listing, to ensure correct
    # tokenization!
    Operator('dot', '.', BINARY_OP, 1),
    Operator('arrow', '->', BINARY_OP, 1),
    Operator('call', None, BINARY_OP, 1), # ()
    Operator('index', None, BINARY_OP, 1), # []
    Operator('add', '+', BINARY_OP, 4),
    Operator('sub', '-', BINARY_OP, 4),
    Operator('shl', '<<', BINARY_OP, 5),
    Operator('shr', '>>', BINARY_OP, 5),
    Operator('not', '~', UNARY_OP, 2, True),
    Operator('mul', '*', BINARY_OP, 3),
    Operator('div', '/', BINARY_OP, 3),
    Operator('mod', '%', BINARY_OP, 3),
    Operator('pos', '+', UNARY_OP, 2, True),
    Operator('neg', '-', UNARY_OP, 2, True),
    Operator('eq', '==', BINARY_OP, 7),
    Operator('ne', '!=', BINARY_OP, 7),
    Operator('le', '<=', BINARY_OP, 6),
    Operator('ge', '>=', BINARY_OP, 6),
    Operator('lt', '<', BINARY_OP, 6),
    Operator('gt', '>', BINARY_OP, 6),
    Operator('lognot', '!', UNARY_OP, 2, True),
    Operator('logand', '&&', BINARY_OP, 11),
    Operator('logor', '||', BINARY_OP, 12),
    Operator('and', '&', BINARY_OP, 8),
    Operator('or', '|', BINARY_OP, 9),
    Operator('xor', '^', BINARY_OP, 10),
    Operator('deref', '*', UNARY_OP, 2, True),
    Operator('addr', '&', UNARY_OP, 2, True),
    Operator('sizeof', 'sizeof', UNARY_OP, 2, True),
    Operator('assign', '=', BINARY_OP, 14, True),
)
TYPE_EXPR_OPS = {'deref'}
OPS_BY_NAME = {op.name: op for op in OPS}
UNOPS_BY_TOKEN = {op.token: op for op in OPS if op.token and op.arity == UNARY_OP}
BINOPS_BY_TOKEN = {op.token: op for op in OPS if op.token and op.arity == BINARY_OP}

KEYWORDS = (
    'if',
    'else',
    'while',
    'break',
    'continue',
    'do',
    'for',
    'struct',
    'union',
    'enum',
    'typedef',
    'return',
)

TOKEN_PATTERNS = {
    'comment': r'//.*',
    'num': r'-?[0-9]+',
    'hex': r'0x[0-9a-fA-F]+',
    'ptr': r'&0x[0-9a-fA-F]+',
    'str': r'"(?:[^"]|\\")*"',
    'directive': r'#[_a-zA-Z]+',
    'keyword': '|'.join(KEYWORDS),
    'type': '|'.join(BASIC_TYPES),
    'name': r'[_a-zA-Z][_a-zA-Z0-9]*',
    'op': '|'.join(re.escape(op.token) for op in OPS if op.token),
    'special': r'[()\[\]{},;]',
}
TOKEN_REGEX = re.compile('|'.join(
    f'(?P<{name}>{pat})' for name, pat in TOKEN_PATTERNS.items()))

ATOM_TOKENS = ('name', 'num', 'hex', 'ptr', 'str')


class ParseError(Exception):
    pass

class Token(NamedTuple):
    toktype: str
    token: str
    filename: str
    row: int
    col: int

def tokenize(text: str, filename: str = '<NO FILE>') -> List[Token]:
    """

        >>> text = '''#include "some file" // a comment!
        ... if (x == 3) {
        ...     break
        ... }'''
        >>> for token in tokenize(text, 'test.c'): print(token)
        Token(toktype='directive', token='#include', filename='test.c', row=1, col=1)
        Token(toktype='str', token='"some file"', filename='test.c', row=1, col=10)
        Token(toktype='comment', token='// a comment!', filename='test.c', row=1, col=22)
        Token(toktype='keyword', token='if', filename='test.c', row=2, col=1)
        Token(toktype='special', token='(', filename='test.c', row=2, col=4)
        Token(toktype='name', token='x', filename='test.c', row=2, col=5)
        Token(toktype='op', token='==', filename='test.c', row=2, col=7)
        Token(toktype='num', token='3', filename='test.c', row=2, col=10)
        Token(toktype='special', token=')', filename='test.c', row=2, col=11)
        Token(toktype='special', token='{', filename='test.c', row=2, col=13)
        Token(toktype='keyword', token='break', filename='test.c', row=3, col=5)
        Token(toktype='special', token='}', filename='test.c', row=4, col=1)
        Token(toktype='eof', token='', filename='test.c', row=4, col=2)

    """
    lines = text.splitlines()
    if not lines:
        lines.append('')
    for row, line in enumerate(lines, 1):
        for match in TOKEN_REGEX.finditer(line):
            yield Token(
                toktype=match.lastgroup,
                token=match.group(),
                filename=filename,
                row=row,
                col=match.start() + 1,
            )
    last_line = lines[-1]
    yield Token(
        toktype='eof',
        token='',
        filename=filename,
        row=len(lines),
        col=len(last_line) + 1,
    )

class TokenIterator:
    """

        >>> it = TokenIterator(tokenize('1 + 2'))
        >>> it.get_next()
        ('num', '1')
        >>> it.get_next()
        ('op', '+')
        >>> with it: raise Exception("BOOM!")
        Traceback (most recent call last):
         ...
        tinc.ParseError: In file <NO FILE>, row 1, col 3 ('+'): BOOM!

        >>> it.get_next()
        ('num', '2')
        >>> it.unget()
        >>> with it: raise Exception("BOOM!")
        Traceback (most recent call last):
         ...
        tinc.ParseError: In file <NO FILE>, row 1, col 5 ('2'): BOOM!

        >>> it.get_next()
        ('num', '2')

    """

    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.n_tokens = len(self.tokens)
        self.i = 0
        self.i2 = 0

    def __enter__(self):
        return self

    def __exit__(self, ex_type, ex, tb):
        if ex is not None and not isinstance(ex, ParseError):
            prev_i = self.i2 - 1
            if prev_i >= 0 and prev_i < self.n_tokens:
                err_token = self.tokens[prev_i]
                raise ParseError(f"In file {err_token.filename}, row {err_token.row}, col {err_token.col} ({err_token.token!r}): {ex}")
            else:
                raise ParseError(f"In ???: {ex}")

    def get_next(self):
        if self.i >= self.n_tokens:
            raise Exception("End of file!")
        token = self.tokens[self.i]
        self.i += 1
        self.i2 = self.i
        return token.toktype, token.token

    def unget(self):
        if self.i <= 0:
            raise IndexError
        self.i -= 1


def invert_type_expr(type_expr, base_type):
    if type_expr == 'base':
        return base_type
    tag = type_expr[0]
    if tag == 'deref':
        return invert_type_expr(type_expr[1], ('ptr', base_type))
    elif tag == 'index':
        return invert_type_expr(type_expr[1], ('array', type_expr[2], base_type))
    elif tag == 'call':
        return invert_type_expr(type_expr[1], ('func', base_type, type_expr[2]))
    else:
        raise Exception(f"Unrecognized type expression tag: {tag!r}")

class TincParser:
    """The TINC language (think "Tiny C") compiles to the assembly language
    used by the TinyProcessor."""

    def __init__(self, tokens):
        if not isinstance(tokens, TokenIterator):
            tokens = TokenIterator(tokens)
        self.tokens = tokens
        self.typedefs = {}
        self.structlike_defs = {
            'struct': {},
            'union': {},
            'enum': {},
        }

        self.get_next = tokens.get_next
        self.unget = tokens.unget

    def expect(self, expected):
        toktype, token = self.get_next()
        if token != expected:
            raise Exception(f"Expected {expected!r}")

    def is_type(self, toktype: str, token: str) -> bool:
        return (
            toktype == 'type'
            or token in ('struct', 'union', 'enum')
            or token in self.typedefs)

    def parse_expr(self, prev_op: Operator = None):
        """Parse an expression.

            >>> from pprint import pprint
            >>> test = lambda text: pprint(TincParser(tokenize(text)).parse_expr())

            >>> test(')')
            None

            >>> test('1 + (2 + #define)')
            Traceback (most recent call last):
             ...
            tinc.ParseError: In file <NO FILE>, row 1, col 10 ('#define'): Expected an expression

            >>> test('1 + 2 + 3')
            ('add',
             ('add', ('atom', 'num', '1'), ('atom', 'num', '2')),
             ('atom', 'num', '3'))

            >>> test('x = y = z')
            ('assign',
             ('atom', 'name', 'x'),
             ('assign', ('atom', 'name', 'y'), ('atom', 'name', 'z')))

            >>> test('1 * 2 + 3')
            ('add',
             ('mul', ('atom', 'num', '1'), ('atom', 'num', '2')),
             ('atom', 'num', '3'))

            >>> test('1 + 2 * 3')
            ('add',
             ('atom', 'num', '1'),
             ('mul', ('atom', 'num', '2'), ('atom', 'num', '3')))

            >>> test('(1 * - 2) + f(3, 4)[5]')
            ('add',
             ('mul', ('atom', 'num', '1'), ('neg', ('atom', 'num', '2'))),
             ('index',
              ('call', ('atom', 'name', 'f'), [('atom', 'num', '3'), ('atom', 'num', '4')]),
              ('atom', 'num', '5')))

            >>> test('{10, 20, 30}[i]')
            ('index',
             ('array',
              [('atom', 'num', '10'), ('atom', 'num', '20'), ('atom', 'num', '30')]),
             ('atom', 'name', 'i'))

            >>> test('*x[3]')
            ('deref', ('index', ('atom', 'name', 'x'), ('atom', 'num', '3')))

            >>> test('(*x)[3])')
            ('index', ('deref', ('atom', 'name', 'x')), ('atom', 'num', '3'))

            >>> test('f()')
            ('call', ('atom', 'name', 'f'), [])

            >>> test('f(1, 2)')
            ('call', ('atom', 'name', 'f'), [('atom', 'num', '1'), ('atom', 'num', '2')])

            >>> test('*f()')
            ('deref', ('call', ('atom', 'name', 'f'), []))

            >>> test('(*f)()')
            ('call', ('deref', ('atom', 'name', 'f')), [])

        """
        with self.tokens:
            toktype, token = self.get_next()
            if toktype == 'op':
                op = UNOPS_BY_TOKEN.get(token)
                if op is None:
                    raise Exception("Not a unary operator")
                child = self.parse_expr(op)
                if child is None:
                    raise Exception("Expected an expression")
                lhs = (op.name, child)
            elif token == '(':
                child = self.parse_expr()
                if child is None:
                    raise Exception("Expected an expression")
                self.expect(')')
                lhs = child
            elif token == '{':
                children = []
                while True:
                    child = self.parse_expr()
                    if child is None:
                        break
                    children.append(child)
                    toktype, token = self.get_next()
                    if token != ',':
                        self.unget()
                        break
                self.expect('}')
                lhs = ('array', children)
            elif toktype in ('name', 'num', 'hex', 'ptr', 'str'):
                lhs = ('atom', toktype, token)
            else:
                self.unget()
                return None

            while True:
                toktype, token = self.get_next()
                if toktype == 'op':
                    op = BINOPS_BY_TOKEN.get(token)
                    if op is None:
                        raise Exception("Not a binary operator")
                    if not prec_check(prev_op, op):
                        self.unget()
                        break
                    rhs = self.parse_expr(op)
                    if rhs is None:
                        raise Exception("Expected an expression")
                    lhs = (op.name, lhs, rhs)
                elif token == '(':
                    op = OPS_BY_NAME['call']
                    if not prec_check(prev_op, op):
                        self.unget()
                        break
                    args = []
                    while True:
                        arg = self.parse_expr()
                        if arg is None:
                            break
                        args.append(arg)
                        toktype, token = self.get_next()
                        if token != ',':
                            self.unget()
                            break
                    self.expect(')')
                    lhs = (op.name, lhs, args)
                elif token == '[':
                    op = OPS_BY_NAME['index']
                    if not prec_check(prev_op, op):
                        self.unget()
                        break
                    rhs = self.parse_expr()
                    if rhs is None:
                        raise Exception("Expected an expression")
                    self.expect(']')
                    lhs = (op.name, lhs, rhs)
                else:
                    self.unget()
                    break

            return lhs

    def parse_type_expr(self, prev_op: Operator = None):
        """Parse a type expression.
        Returns a tuple (name, type_expr) or None.
        The name may be None.
        The type_expr is either base_type, or a more complicated expression
        built on top of it.

            >>> from pprint import pprint
            >>> test = lambda text: pprint(TincParser(tokenize(text)).parse_type_expr())

            >>> test('')
            (None, 'base')

            >>> test('x')
            ('x', 'base')

            >>> test('*x')
            ('x', ('deref', 'base'))

            >>> test('(*x)')
            ('x', ('deref', 'base'))

            >>> test('(*)')
            (None, ('deref', 'base'))

            >>> test('x[3]')
            ('x', ('index', 'base', 3))

            >>> test('[3]')
            (None, ('index', 'base', 3))

            >>> test('*x[3]')
            ('x', ('deref', ('index', 'base', 3)))

            >>> test('*[3]')
            (None, ('deref', ('index', 'base', 3)))

            >>> test('**[3]')
            (None, ('deref', ('deref', ('index', 'base', 3))))

            >>> test('f()')
            ('f', ('call', 'base', []))

            >>> test('f(int x, int y)')
            ('f', ('call', 'base', [('x', ('basic', 'int')), ('y', ('basic', 'int'))]))

            >>> test('*f(int x)')
            ('f', ('deref', ('call', 'base', [('x', ('basic', 'int'))])))

            >>> test('(*f)(int x)')
            ('f', ('call', ('deref', 'base'), [('x', ('basic', 'int'))]))

            >>> test('f(int x, int x)')
            Traceback (most recent call last):
             ...
            tinc.ParseError: In file <NO FILE>, row 1, col 15 (')'): Duplicate function parameter name: 'x'

        """
        with self.tokens:
            toktype, token = self.get_next()
            if toktype == 'op':
                op = UNOPS_BY_TOKEN.get(token)
                if op is None:
                    raise Exception("Not a unary operator")
                if op.name not in TYPE_EXPR_OPS:
                    raise Exception("Operator not allowed in type expressions")
                child = self.parse_type_expr(op)
                if child is None:
                    raise Exception("Expected a type expression")
                name, child_expr = child
                child_expr = (op.name, child_expr)
            elif token == '(':
                child = self.parse_type_expr()
                if child is None:
                    raise Exception("Expected a type expression")
                self.expect(')')
                name, child_expr = child
            elif toktype == 'name':
                name = token
                child_expr = 'base'
            else:
                name = None
                child_expr = 'base'
                self.unget()

            while True:
                toktype, token = self.get_next()
                if token == '(':
                    op = OPS_BY_NAME['call']
                    if not prec_check(prev_op, op):
                        self.unget()
                        break
                    args = []
                    arg_names = set()
                    while True:
                        arg_decl = self.parse_decl()
                        if arg_decl is None:
                            break
                        arg_name = arg_decl[0]
                        if arg_name in arg_names:
                            raise Exception(f"Duplicate function parameter name: {arg_name!r}")
                        arg_names.add(arg_name)
                        args.append(arg_decl)
                        toktype, token = self.get_next()
                        if token != ',':
                            self.unget()
                            break
                    self.expect(')')
                    child_expr = (op.name, child_expr, args)
                elif token == '[':
                    op = OPS_BY_NAME['index']
                    if not prec_check(prev_op, op):
                        self.unget()
                        break
                    index = None
                    toktype, token = self.get_next()
                    if toktype == 'num':
                        index = int(token)
                    else:
                        self.unget()
                    self.expect(']')
                    child_expr = (op.name, child_expr, index)
                else:
                    self.unget()
                    break

            return name, child_expr

    def parse_decl(self, toplevel=False):
        """Parse a variable declaration.
        Returns a tuple (name, type_expr), or None.
        The name may be None.

            >>> from pprint import pprint
            >>> test = lambda text: pprint(TincParser(tokenize(text)).parse_decl())

            >>> test(')')
            None

            >>> test('int')
            (None, ('basic', 'int'))

            >>> test('int *')
            (None, ('ptr', ('basic', 'int')))

            >>> test('int x')
            ('x', ('basic', 'int'))

            >>> test('int *x')
            ('x', ('ptr', ('basic', 'int')))

            >>> test('int **x')
            ('x', ('ptr', ('ptr', ('basic', 'int'))))

            >>> test('int *x[3]')
            ('x', ('array', 3, ('ptr', ('basic', 'int'))))

            >>> test('struct point *p')
            ('p', ('ptr', ('struct', 'point', None)))

            >>> test('struct point { int x; int y; } *p')
            ('p',
             ('ptr',
              ('struct', 'point', [('x', ('basic', 'int')), ('y', ('basic', 'int'))])))

            >>> test('struct { int x; int y; } *p')
            ('p',
             ('ptr', ('struct', None, [('x', ('basic', 'int')), ('y', ('basic', 'int'))])))

            >>> test('enum E {x, y = 1, z}')
            (None, ('enum', 'E', [('x', None), ('y', 1), ('z', None)]))

        """
        with self.tokens:
            toktype, token = self.get_next()
            if toktype == 'type':
                base_type = ('basic', token)
            elif token in ('struct', 'union', 'enum'):
                tag = token
                name = None
                children = None
                toktype, token = self.get_next()
                if toktype == 'name':
                    name = token
                    toktype, token = self.get_next()
                    if name in self.structlike_defs[tag]:
                        raise Exception(f"Redefinition of {tag} {name!r}")
                if token == '{':
                    children = []
                    if tag == 'enum':
                        while True:
                            toktype, token = self.get_next()
                            if toktype != 'name':
                                self.unget()
                                break
                            child_name = token
                            default = None
                            toktype, token = self.get_next()
                            if token == '=':
                                toktype, token = self.get_next()
                                if toktype != 'num':
                                    raise Exception("Expected a number")
                                default = int(token)
                            else:
                                self.unget()
                            children.append((child_name, default))
                            toktype, token = self.get_next()
                            if token != ',':
                                self.unget()
                                break
                    else:
                        while True:
                            child = self.parse_decl()
                            if child is None:
                                break
                            self.expect(';')
                            children.append(child)
                    self.expect('}')
                    if name is not None:
                        self.structlike_defs[tag] = children
                else:
                    self.unget()
                if name is None and children is None:
                    raise Exception("Expected a name or '{'")
                base_type = (tag, name, children)
            elif token in self.typedefs:
                base_type = ('typedef', token)
            else:
                self.unget()
                return None

            type_expr = self.parse_type_expr()
            if type_expr is None:
                raise Exception("Expected a type expression")
            name, type_expr = type_expr
            type_expr = invert_type_expr(type_expr, base_type)

            return (name, type_expr)

    def parse_statement(self):
        """Parse a statement.

            >>> from pprint import pprint
            >>> test = lambda text: pprint(TincParser(tokenize(text)).parse_statement())

            >>> test(')')
            None

            >>> test('1 + 2')
            Traceback (most recent call last):
             ...
            tinc.ParseError: In file <NO FILE>, row 1, col 6 (''): Expected ';'

            >>> test('1 + 2;')
            ('expr', ('add', ('atom', 'num', '1'), ('atom', 'num', '2')))

            >>> test('int x;')
            ('decl', 'x', ('basic', 'int'), None)

            >>> test('int x = 3;')
            ('decl', 'x', ('basic', 'int'), ('atom', 'num', '3'))

            >>> test('{ int x = 3; f(&x); }')
            ('block',
             [('decl', 'x', ('basic', 'int'), ('atom', 'num', '3')),
              ('expr', ('call', ('atom', 'name', 'f'), [('addr', ('atom', 'name', 'x'))]))])

            >>> test('if (x == 3) { break; }')
            ('if',
             ('eq', ('atom', 'name', 'x'), ('atom', 'num', '3')),
             ('block', [('break',)]),
             None)

            >>> test('if (true) 1; else 2;')
            ('if',
             ('atom', 'name', 'true'),
             ('expr', ('atom', 'num', '1')),
             ('expr', ('atom', 'num', '2')))

            >>> test('while (true) ;')
            ('while', ('atom', 'name', 'true'), ('noop',))

        """
        with self.tokens:
            toktype, token = self.get_next()
            if token == '{':
                statements = []
                while True:
                    child = self.parse_statement()
                    if child is None:
                        break
                    statements.append(child)
                self.expect('}')
                return ('block', statements)
            elif token == 'if':
                self.expect('(')
                cond = self.parse_expr()
                if cond is None:
                    raise Exception("Expected an expression")
                self.expect(')')
                if_branch = self.parse_statement()
                if if_branch is None:
                    raise Exception("Expected a statement")
                toktype, token = self.get_next()
                if token == 'else':
                    else_branch = self.parse_statement()
                    if else_branch is None:
                        raise Exception("Expected a statement")
                else:
                    else_branch = None
                    self.unget()
                return ('if', cond, if_branch, else_branch)
            elif token == 'while':
                self.expect('(')
                cond = self.parse_expr()
                if cond is None:
                    raise Exception("Expected an expression")
                self.expect(')')
                child = self.parse_statement()
                if child is None:
                    raise Exception("Expected a statement")
                return ('while', cond, child)
            elif self.is_type(toktype, token):
                self.unget()
                name, child = self.parse_decl()
                toktype, token = self.get_next()
                if token == '=':
                    default = self.parse_expr()
                    if default is None:
                        raise Exception("Expected an expression")
                else:
                    default = None
                    self.unget()
                self.expect(';')
                return ('decl', name, child, default)
            elif token == 'return':
                child = self.parse_expr()
                if child is None:
                    raise Exception("Expected an expression")
                self.expect(';')
                return ('return', child)
            elif token == ';':
                return ('noop',)
            elif token in ('break', 'continue'):
                self.expect(';')
                return (token,)
            else:
                self.unget()
                expr = self.parse_expr()
                if expr is not None:
                    self.expect(';')
                    return ('expr', expr)
                else:
                    return None

    def parse_toplevel(self):
        """Parse a top-level statement.

            >>> from pprint import pprint
            >>> test = lambda text: pprint(TincParser(tokenize(text)).parse_toplevel())

            >>> test('+')
            None

            >>> test(';')
            ('noop',)

            >>> test('typedef int *i;')
            ('typedef', 'i', ('ptr', ('basic', 'int')))

            >>> test('typedef int adder(int);')
            ('typedef', 'adder', ('func', ('basic', 'int'), [(None, ('basic', 'int'))]))

            >>> test('int adder(int x) return i + 1;')
            ('def',
             'adder',
             ('basic', 'int'),
             {'x': ('basic', 'int')},
             ('return', ('add', ('atom', 'name', 'i'), ('atom', 'num', '1'))))

            >>> test('int adder return i + 1;')
            Traceback (most recent call last):
             ...
            tinc.ParseError: In file <NO FILE>, row 1, col 11 ('return'): Expected '=' or ';'

            >>> test('int adder(int) return i + 1;')
            Traceback (most recent call last):
             ...
            tinc.ParseError: In file <NO FILE>, row 1, col 16 ('return'): Function parameter 1 needs a name

        """
        with self.tokens:
            toktype, token = self.get_next()
            if self.is_type(toktype, token):
                self.unget()
                name, type_expr = self.parse_decl()
                toktype, token = self.get_next()
                if token == '=':
                    default = self.parse_expr()
                    if default is None:
                        raise Exception("Expected an expression")
                else:
                    default = None
                    self.unget()
                toktype, token = self.get_next()
                if token == ';':
                    return ('decl', name, type_expr, default)
                elif default is not None:
                    raise Exception("Expected ';'")
                elif type_expr[0] != 'func':
                    raise Exception("Expected '=' or ';'")
                else:
                    ret_expr = type_expr[1]
                    args = type_expr[2]
                    for i, (arg_name, arg_expr) in enumerate(args, 1):
                        if arg_name is None:
                            raise Exception(f"Function parameter {i} needs a name")
                    args = dict(args)
                    self.unget()
                    statement = self.parse_statement()
                    if statement is None:
                        raise Exception("Expected a statement")
                    return ('def', name, ret_expr, args, statement)
            elif token == 'typedef':
                type_expr = self.parse_decl()
                if type_expr is None:
                    raise Exception("Expected a declaration")
                name, type_expr = type_expr
                if name is None:
                    raise Exception("typedefs require a name")
                if name in self.typedefs:
                    raise Exception(f"Redefinition of typedef {name!r}")
                self.expect(';')
                self.typedefs[name] = type_expr
                return ('typedef', name, type_expr)
            elif token == ';':
                return ('noop',)
            else:
                self.unget()
                return None

    def parse(self):
        toplevels = []
        with self.tokens:
            while True:
                toplevel = self.parse_toplevel()
                if toplevel is None:
                    break
                toplevels.append(toplevel)
            toktype, token = self.get_next()
            if toktype != 'eof':
                raise Exception("Expected end of file!")
        return toplevels


def parse_cli_args():
    parser = ArgumentParser()
    parser.add_argument('--file', '-f', required=True)
    return parser.parse_args()


def main(args=None):
    if args is None:
        args = parse_cli_args()
    if args.file == '-':
        filename = '<STDIN>'
        file = sys.stdin
    else:
        filename = args.file
        file = open(filename, 'r')
    text = file.read()
    file.close()
    parser = TincParser(tokenize(text, filename))
    toplevels = parser.parse()
    for toplevel in toplevels:
        print(toplevel)


if __name__ == '__main__':
    main()
