
typedef struct A {
    struct A *next;
} A;

typedef int adder(int);
typedef int (*adder_ptr)(int i);

adder add1;

int add1(int i) return i + 1;

int main() {
    adder_ptr p = &add1;
    return 0;
}
