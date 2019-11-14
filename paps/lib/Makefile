CC = gcc

default: libfunctions.a

libfunctions.a: functions.o
	ar rcs $@ $^

functions.o: functions.c functions.h
	$(CC) -c $<

clean:
	rm *.o *.a
