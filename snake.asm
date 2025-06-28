JMP main


%macro CALL D=PC D+ 10 JMP
%macro RET PC=D


%define screen_w 40
%define screen_h 20
%define screen_size ( screen_w * screen_h )
%define max_snake_length 20
%define dot_offset -3
%define dot_freq 31

%define snake_char "S"
%define dot_char "."

"SCR:"
screen: %zeros chr screen_size
"TAIL:"
; array of pointers into screen
; the first entry is equivalent to snake_x, snake_y
tail: %zeros ptr max_snake_length
"SNAKE:"
snake_x: 0
snake_y: 0
snake_length: 1


"IS:"
init_snake:
; set first tail position to snake_x, snake_y
A=: snake_y
A* screen_w
A+: snake_x
A+ screen
:=A tail
RET


"IS:"
init_screen:

; fill screen with empty space
A=0
B= screen
_loop:
C=A C+ dot_offset C% dot_freq C?0 JNE _no_dot
.B= dot_char
JMP _endif
_no_dot:
.B= " "
_endif:
B+1 A+1
A? screen_size JL _loop

; draw snake's head (i.e. tail[0])
A=: tail .A= snake_char

RET


"US:"
update_snake:
; we have just moved (one of u/d/l/r), so now we need to check whether
; we are eating (i.e. have collided with) a dot, update our tail, and
; render the snake (i.e. possibly erase the tip of its tail, and always
; render its head at the new position)

; check whether we're colliding with something
; (our own body, in which case it's game over, or a dot, in which case
; we eat it)
A=: snake_y
A* screen_w
A+: snake_x
A+ screen
.A? snake_char JE game_over
.A? dot_char JNE _erase_tail_end
:? snake_length max_snake_length JGE _erase_tail_end ; eat without growing

; we are eating a dot, so increase our length, and don't
; erase the tip of our tail
:+1 snake_length
JMP _update_tail

_erase_tail_end:
; we are not eating a dot, so erase the tip of our tail
; erase *tail[snake_length - 1]
A=: snake_length A-1 A*2 A+ tail
A=:A .A= " "

_update_tail:
; update tail positions
A=: snake_length A-1
_loop:
A?0 JLE _update_head
B=A B*2 B+ tail
C=B C-2
:B=:C ; tail[A] = tail[A - 1]
A-1
JMP _loop

_update_head:
; set tail[0] to the head of the snake, i.e. snake_y * screen_w + snake_x
A=: snake_y
A* screen_w
A+: snake_x
A+ screen
:=A tail

; draw snake's head (i.e. tail[0])
A=: tail .A= snake_char

RET


"GAME OVER:"
game_over:
HALT


"PS:"
print_screen:

; print a horizontal line
IO= "+"
A=0
_loop0:
IO= "-"
A+1 A? screen_w JL _loop0
IO= "+" IO= "\n"

; print the contents of the "screen" (i.e. the bytes in that region of memory)
A=0
B= screen
_loop1:
C=A C% screen_w C?0 JNE _loop1_start
IO= "|"
_loop1_start:
IO=.B
B+1 A+1
C=A C% screen_w C?0 JNE _loop1_end
IO= "|" IO= "\n"
_loop1_end:
A? screen_size JL _loop1

; print a horizontal line
IO= "+"
A=0
_loop2:
IO= "-"
A+1 A? screen_w JL _loop2
IO= "+" IO= "\n"

RET


"CK:"
check_keys:
_loop:
A=IO
_check_u: A? "\0u" JNE _check_l
    ; move up
    :?0 snake_y JLE _loop
    :-1 snake_y
    JMP _done
_check_l: A? "\0l" JNE _check_r
    ; move left
    :?0 snake_x JLE _loop
    :-1 snake_x
    JMP _done
_check_r: A? "\0r" JNE _check_d
    ; move right
    :? snake_x ( screen_w - 1 ) JGE _loop
    :+1 snake_x
    JMP _done
_check_d: A? "\0d" JNE _check_q
    ; move down
    :? snake_y ( screen_h - 1 ) JGE _loop
    :+1 snake_y
    JMP _done
_check_q: A? "\0q" JNE _loop
    ; quit
    HALT
_done:
RET


"PM:"
print_msg:
_start:
.A? "\0"
JE _done
IO=.A
A+1
JMP _start
_done:
RET


"M:"
main:
CALL init_snake
CALL init_screen
_loop:
CALL print_screen
A= _input CALL print_msg
CALL check_keys
CALL update_snake
JMP _loop
_input: "Enter u/d/l/r to move, q to quit: \0"
