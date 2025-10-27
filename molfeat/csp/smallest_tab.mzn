include "globals.mzn";

% Parameters
int: size_mol;
int: size_mol2;

array[1..size_mol, 1..size_mol2] of float: differences;

% Variables
var float: smallest_size;
array[1..size_mol] of var int: smallest_idx;

% Constraints
constraint all_different(smallest_idx);
constraint smallest_size = sum(i in 1..size_mol)(differences[i, smallest_idx[i]]);


solve minimize smallest_size;