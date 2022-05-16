QTensor
====

   This library is a python implementation using 
the PyTorch framework for working with quantum states 
in the Tensor Train format.
The library itself is located in the qtensor folder, 
and the examples folder contains typical examples.

Installation
====

**It is recommended** to install the library 
in a virtual environment. You also need to install 
the numpy, scipy, matplotlib, tqdm and torch libraries 
into this virtual environment.

### Linux
```
   python -m venv QTensor
   source QTensor/bin/activate
```

### Windows
```
   python -m venv QTensor
   QTensor\Scripts\activate.bat 
```

## Installing from source code
To install the stable or development version, 
you need to install from the source. 
First, clone the repository:

```
   git clone https://gitlab.com/SergeiKuzmin/qtensor.git
```

```
   python setup.py install
```
What those packages do
====

They have the following functionality:

- `qtensor` : The library itself
- `qtensor._mps` : Implements a quantum state in Tensor Train format
- `qtensor._state` : Implements a quantum state in the usual format
- `qtensor._circuit` : Contains basic quantum circuits
- `qtensor._ham` : Contains the basic hamiltonians

Current maintainer is [Sergei Kuzmin](https://gitlab.com/SergeiKuzmin).
