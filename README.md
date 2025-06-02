# Tiny Computer

TODO: describe this thing
TL;DR: it's a little computer, with memory, an instruction set
and a processor, and an assembly language.
For teaching/learning about such things.

## Motivational example

Given a file with assembly like this:
```
$ cat test.asm
JMP main

"STK:"
%define callstack_size 10
callstack: %zeros callstack_size
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
readline_buf: %zeros readline_buf_size
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
A= _hello CALL printline
A= readline_buf CALL printline
IO= "!"
IO= "\n"
HALT
_prompt: "What is your name? \0"
_hello: "Hello, \0"
```

We can boot it up like so:
```
$ python tinycomp.py -f test.asm --bars -r
What is your name? the world
Hello, the world!
     +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
0000 |  JMP|&0082| S| T| K| :|&00ba|00|00|00|00|00|00|
0010 |00|00| P| L| :|  .A?|\0|   JE|&0024|IO=.A|  A+1|
0020 |  JMP|&0015|PC=:D| R| L| :| t| h| e|  | w| o| r|
0030 | l| d|\0|00|00|00|00|00|00|00|00|00|00|00|00|00|
0040 |00|00|00|00|00|00|00|00|00|00|00|00|00|00|00|00|
0050 |00|00|00|00|00|00|00|00|00|00|00|   A=|&0029|.A=IO|
0060 |<<|  .A?|\n|   JE|&0076|  A+1|   A?|&005b|  JGE|
0070 |&007b|  JMP|&005f|  .A=|\0|PC=:D| HALT| M| A| I|
0080 | N| :|   D=|&0008|   A=|&00c2|:D=PC|  :D+|   10|
0090 |  JMP|&0015|:D=PC|  :D+|   10|  JMP|&005b|   A=|
00a0 |&00d6|:D=PC|  :D+|   10|  JMP|&0015|   A=|&0029|
00b0 |:D=PC|  :D+|   10|  JMP|&0015|  IO=| !|  IO=|\n|
00c0 | HALT| W| h| a| t|  | i| s|  | y| o| u| r|  | n|
00d0 | a| m| e| ?|  |\0| H| e| l| l| o| ,|  |\0|##|##|
     +--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+--+
A=&0032
B=00
C=00
D=&0008
PC=&00c0
CMP=0
```
