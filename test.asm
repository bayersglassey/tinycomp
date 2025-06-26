JMP main

"STK:"
%define callstack_size 5
callstack: %zeros ptr callstack_size
%macro CALL :D=PC :D+ 10 JMP
%macro RET PC=:D

"PL:"
printline:
_start:
.A? "\0"
JE _done
IO=.A
A+1
JMP _start
_done:
RET

"RL:"
%define readline_buf_size 50
readline_buf: %zeros chr readline_buf_size
readline_buf_end:
readline:
A= readline_buf
_start:
.A=IO
.A? "\n"
JE _done
A+1
A? readline_buf_end
JGE _nospace
JMP _start
_done:
.A= "\0"
RET
_nospace:
HALT ; error, we ran out of space in the buffer!

"MAIN:"
main:
D= callstack
A= _prompt CALL printline
CALL readline
hello:
A= _hello CALL printline
A= readline_buf CALL printline
IO= "!"
IO= "\n"
HALT
_prompt: "What is your name? \0"
_hello: "Hello, \0"
