IPATH=-Iinc/
SRC=src/
OBJ=obj/
BIN=bin/

all:app
	@./bin/app

app:main.o matrix.o neuralNetwork.o
	@g++ $(OBJ)* -o $(BIN)app

main.o:
	@g++ $(IPATH) -c $(SRC)main.cpp -o $(OBJ)main.o

matrix.o:
	@g++ $(IPATH) -c $(SRC)matrix.cpp -o $(OBJ)matrix.o

neuralNetwork.o:
	@g++ $(IPATH) -c $(SRC)neuralNetwork.cpp -o $(OBJ)neuralNetwork.o

clean:
	rm -rf $(OBJ)*.o
	rm -rf $(BIN)*