import re
from typing import List, Tuple, Dict, NamedTuple, Any
from collections import deque

from tinycomp import *


UNARY_OP = 'unary'
BINARY_OP = 'binary'

class Operator(NamedTuple):
    name: str
    token: str
    arity: str # UNARY_OP, BINARY_OP, etc
    prec: int # precedence


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
    Operator('add', '+', BINARY_OP, 4),
    Operator('sub', '-', BINARY_OP, 4),
    Operator('shl', '<<', BINARY_OP, 5),
    Operator('shr', '>>', BINARY_OP, 5),
    Operator('mul', '*', BINARY_OP, 3),
    Operator('div', '/', BINARY_OP, 3),
    Operator('mod', '%', BINARY_OP, 3),
    Operator('pos', '+', UNARY_OP, 2),
    Operator('neg', '-', UNARY_OP, 2),
    Operator('eq', '==', BINARY_OP, 7),
    Operator('ne', '!=', BINARY_OP, 7),
    Operator('le', '<=', BINARY_OP, 6),
    Operator('ge', '>=', BINARY_OP, 6),
    Operator('lt', '<', BINARY_OP, 6),
    Operator('gt', '>', BINARY_OP, 6),
    Operator('not', '!', UNARY_OP, 2),
    Operator('logand', '&&', BINARY_OP, 11),
    Operator('logor', '||', BINARY_OP, 12),
    Operator('and', '&', BINARY_OP, 8),
    Operator('or', '|', BINARY_OP, 9),
    Operator('xor', '^', BINARY_OP, 10),
    Operator('deref', '*', UNARY_OP, 2),
    Operator('addr', '&', UNARY_OP, 2),
    Operator('sizeof', 'sizeof', UNARY_OP, 2),
    Operator('assign', '=', BINARY_OP, 14),
)
OPS_BY_NAME = {op.name: op for op in OPS}
UNOPS_BY_TOKEN = {op.token: op for op in OPS if op.arity == UNARY_OP}
BINOPS_BY_TOKEN = {op.token: op for op in OPS if op.arity == BINARY_OP}

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
    'typedef',
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
    'op': '|'.join(re.escape(op.token) for op in OPS),
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


def parse_expr(tokens, prev_prec: int = None):
    """Parse an expression.

        >>> from pprint import pprint
        >>> test = lambda text: pprint(parse_expr(tokenize(text)))

        >>> test(')')
        None

        >>> test('1 + (2 + #define)')
        Traceback (most recent call last):
         ...
        tinc.ParseError: In file <NO FILE>, row 1, col 10 ('#define'): Expected an expression

        >>> test('1 + 2 + 3')
        ('binop',
         'add',
         ('binop', 'add', ('atom', 'num', '1'), ('atom', 'num', '2')),
         ('atom', 'num', '3'))

        >>> test('1 * 2 + 3')
        ('binop',
         'add',
         ('binop', 'mul', ('atom', 'num', '1'), ('atom', 'num', '2')),
         ('atom', 'num', '3'))

        >>> test('1 + 2 * 3')
        ('binop',
         'add',
         ('atom', 'num', '1'),
         ('binop', 'mul', ('atom', 'num', '2'), ('atom', 'num', '3')))

        >>> test('(1 * - 2) + f(3, 4)[5]')
        ('binop',
         'add',
         ('binop', 'mul', ('atom', 'num', '1'), ('unop', 'neg', ('atom', 'num', '2'))),
         ('index',
          ('call', ('atom', 'name', 'f'), [('atom', 'num', '3'), ('atom', 'num', '4')]),
          ('atom', 'num', '5')))

        >>> test('{10, 20, 30}[i]')
        ('index',
         ('array',
          [('atom', 'num', '10'), ('atom', 'num', '20'), ('atom', 'num', '30')]),
         ('atom', 'name', 'i'))

    """
    def expect(expected):
        toktype, token = tokens.get_next()
        if token != expected:
            raise Exception(f"Expected {expected!r}")
    if not isinstance(tokens, TokenIterator):
        tokens = TokenIterator(tokens)
    with tokens:
        toktype, token = tokens.get_next()
        if toktype == 'op':
            op = UNOPS_BY_TOKEN.get(token)
            if op is None:
                raise Exception(f"Not a unary operator: {token!r}")
            child = parse_expr(tokens, op.prec)
            if child is None:
                raise Exception("Expected an expression")
            lhs = ('unop', op.name, child)
        elif token == '(':
            child = parse_expr(tokens)
            if child is None:
                raise Exception("Expected an expression")
            expect(')')
            lhs = child
        elif token == '{':
            children = []
            while True:
                child = parse_expr(tokens)
                if child is None:
                    break
                children.append(child)
                toktype, token = tokens.get_next()
                if token != ',':
                    tokens.unget()
                    break
            expect('}')
            lhs = ('array', children)
        elif toktype in ('name', 'num', 'hex', 'ptr', 'str'):
            lhs = ('atom', toktype, token)
        else:
            tokens.unget()
            return None

        while True:
            toktype, token = tokens.get_next()
            if toktype == 'op':
                op = BINOPS_BY_TOKEN.get(token)
                if op is None:
                    raise Exception(f"Not a binary operator: {token!r}")
                if prev_prec is not None and op.prec >= prev_prec:
                    tokens.unget()
                    break
                rhs = parse_expr(tokens, op.prec)
                if rhs is None:
                    raise Exception("Expected an expression")
                lhs = ('binop', op.name, lhs, rhs)
            elif token == '(':
                args = []
                while True:
                    arg = parse_expr(tokens)
                    if arg is None:
                        break
                    args.append(arg)
                    toktype, token = tokens.get_next()
                    if token != ',':
                        tokens.unget()
                        break
                expect(')')
                lhs = ('call', lhs, args)
            elif token == '[':
                rhs = parse_expr(tokens)
                if rhs is None:
                    raise Exception("Expected an expression")
                expect(']')
                lhs = ('index', lhs, rhs)
            else:
                tokens.unget()
                break

        return lhs

def parse_decl(tokens):
    """Parse a variable declaration.

        >>> from pprint import pprint
        >>> test = lambda text: pprint(parse_decl(tokenize(text)))

        >>> test(')')
        None

        >>> test('int x')
        ('x', ('basic', 'int'))

        >>> test('int *x')
        ('x', ('ptr', ('basic', 'int')))

        >>> test('int **x')
        ('x', ('ptr', ('ptr', ('basic', 'int'))))

        >>> test('int *x[3]')
        ('x', ('array', 3, ('ptr', ('basic', 'int'))))

        >>> test('struct point *p')
        ('p', ('ptr', ('struct', 'point')))

    """
    def expect(expected):
        toktype, token = tokens.get_next()
        if token != expected:
            raise Exception(f"Expected {expected!r}")
    if not isinstance(tokens, TokenIterator):
        tokens = TokenIterator(tokens)
    with tokens:
        toktype, token = tokens.get_next()
        if toktype == 'type':
            decl_type = ('basic', token)
        elif token in ('struct', 'union'):
            prev_token = token
            toktype, token = tokens.get_next()
            if toktype != 'name':
                raise Exception("Expected a name")
            decl_type = (prev_token, token)
        else:
            tokens.unget()
            return None

        ptr_depth = 0
        while True:
            toktype, token = tokens.get_next()
            if token == '*':
                ptr_depth += 1
            else:
                tokens.unget()
                break

        toktype, token = tokens.get_next()
        if toktype != 'name':
            raise Exception("Expected a name")
        name = token

        for i in range(ptr_depth):
            decl_type = ('ptr', decl_type)

        toktype, token = tokens.get_next()
        if token == '[':
            toktype, token = tokens.get_next()
            if toktype != 'num':
                raise Exception("Expected a number")
            index = int(token)
            expect(']')
            decl_type = ('array', index, decl_type)
        else:
            tokens.unget()

        return (name, decl_type)

def parse_statement(tokens):
    """Parse a statement.

        >>> from pprint import pprint
        >>> test = lambda text: pprint(parse_statement(tokenize(text)))

        >>> test(')')
        None

        >>> test('1 + 2')
        Traceback (most recent call last):
         ...
        tinc.ParseError: In file <NO FILE>, row 1, col 6 (''): Expected ';'

        >>> test('1 + 2;')
        ('expr', ('binop', 'add', ('atom', 'num', '1'), ('atom', 'num', '2')))

        >>> test('int x;')
        ('decl', 'x', ('basic', 'int'), None)

        >>> test('int x = 3;')
        ('decl', 'x', ('basic', 'int'), ('atom', 'num', '3'))

        >>> test('{ int x = 3; f(&x); }')
        ('block',
         [('decl', 'x', ('basic', 'int'), ('atom', 'num', '3')),
          ('expr',
           ('call', ('atom', 'name', 'f'), [('unop', 'addr', ('atom', 'name', 'x'))]))])

        >>> test('if (x == 3) { break; }')
        ('if',
         ('binop', 'eq', ('atom', 'name', 'x'), ('atom', 'num', '3')),
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
    def expect(expected):
        toktype, token = tokens.get_next()
        if token != expected:
            raise Exception(f"Expected {expected!r}")
    if not isinstance(tokens, TokenIterator):
        tokens = TokenIterator(tokens)
    with tokens:
        toktype, token = tokens.get_next()
        if token == '{':
            statements = []
            while True:
                child = parse_statement(tokens)
                if child is None:
                    break
                statements.append(child)
            expect('}')
            return ('block', statements)
        elif token == 'if':
            expect('(')
            cond = parse_expr(tokens)
            if cond is None:
                raise Exception("Expected an expression")
            expect(')')
            if_branch = parse_statement(tokens)
            if if_branch is None:
                raise Exception("Expected a statement")
            toktype, token = tokens.get_next()
            if token == 'else':
                else_branch = parse_statement(tokens)
                if else_branch is None:
                    raise Exception("Expected a statement")
            else:
                else_branch = None
                tokens.unget()
            return ('if', cond, if_branch, else_branch)
        elif token == 'while':
            expect('(')
            cond = parse_expr(tokens)
            if cond is None:
                raise Exception("Expected an expression")
            expect(')')
            child = parse_statement(tokens)
            if child is None:
                raise Exception("Expected a statement")
            return ('while', cond, child)
        elif toktype == 'type' or token in ('struct', 'union'):
            tokens.unget()
            name, child = parse_decl(tokens)
            toktype, token = tokens.get_next()
            if token == '=':
                default = parse_expr(tokens)
                if default is None:
                    raise Exception("Expected an expression")
            else:
                default = None
                tokens.unget()
            expect(';')
            return ('decl', name, child, default)
        elif token == ';':
            return ('noop',)
        elif token in ('break', 'continue'):
            expect(';')
            return (token,)
        else:
            tokens.unget()
            expr = parse_expr(tokens)
            if expr is not None:
                expect(';')
                return ('expr', expr)
            else:
                return None


class TinyCompiler:
    """The TINC language (think "Tiny C") compiles to the assembly language
    used by the TinyProcessor."""

    def __init__(self):
        pass
