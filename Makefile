main.exe: main.o neural_net.o
	g++ -o main.exe main.o neural_net.o

main.o: main.cpp
	g++ -c main.cpp

neural_net.o: neural_net.cpp neural_net.h
	g++ -c neural_net.cpp

debug:
	g++ -g -o mainNeuralDebug.exe main.cpp neural_net.cpp

clean:
	rm -f *.exe *.o *.stackdump *~

backup:
	test -d backups || mkdir backups
	cp *.cpp backups
	cp *.h backups
	cp Makefile backups
