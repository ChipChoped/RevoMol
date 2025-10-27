% Includes
include "globals.mzn";
include "matrix_functions.mzn";

% Parameters
int: natoms; % number of atoms
int: nsingle_bonds; % number of single bonds
int: ndouble_bonds; % number of double bonds
int: ntriple_bonds; % number of triple bonds

array[1..natoms, 1..nsingle_bonds] of int: single_bond;
array[1..natoms, 1..ndouble_bonds] of int: double_bond;
array[1..natoms, 1..ntriple_bonds] of int: triple_bond;
array[1..natoms, 1..3] of int: ring_size;
array[1..natoms] of int: valence;
array[1..natoms] of int: periodic;

function bool: are_identical(int: i, int: j) =
  periodic[i] = periodic[j] /\
  valence[i] = valence[j] /\
  row(single_bond,i) = row(single_bond,j) /\
  row(double_bond,i) = row(double_bond,j) /\
  row(triple_bond,i) = row(triple_bond,j);

% Variables
% Adjacency matrix:  Store atom pairs and bond type.
array[1..natoms, 1..natoms] of var 0..3: adjacency_matrix;

% Bond counters for each atom
array[1..natoms] of var set of 1..natoms: sb;  % Single bond indicators
array[1..natoms] of var set of 1..natoms: db;  % Double bond indicators
array[1..natoms] of var set of 1..natoms: tb;  % Triple bond indicators

% Ensure the rings have the correct size
array[1..natoms, 1..natoms] of var 0..1: ring_matrix;

% Constraints

% Constraints 1: Symmetry of the adjacency matrix (undirected graph)
constraint forall(i, j in 1..natoms where i < j)(
    adjacency_matrix[i, j] = adjacency_matrix[j, i]
);

% Constraints 2:  No self-loops
constraint forall(i in 1..natoms)(
    adjacency_matrix[i, i] = 0
);

% Constraints 3 : Fill the adjacency matrix with the bond type
constraint forall(i, j in 1..natoms)(
  if j in sb[i] then
    valence[j] in row(single_bond, i ) /\
    adjacency_matrix[i, j] = 1
  elseif j in db[i] then
    valence[j] in row(double_bond, i) /\
    adjacency_matrix[i, j] = 2
  elseif j in tb[i] then
    valence[j] in row(triple_bond, i) /\
    adjacency_matrix[i, j] = 3
  else
    adjacency_matrix[i, j] = 0
  endif
);

% Constraints 4: Ensure bond indicators are consistent with bond arrays
constraint forall(i in 1..natoms, j in 1..natoms)(
  (adjacency_matrix[i, j] = 1) <-> (j in sb[i])
);
constraint forall(i in 1..natoms, j in 1..natoms)(
  (adjacency_matrix[i, j] = 2) <-> (j in db[i])
);
constraint forall(i in 1..natoms, j in 1..natoms)(
  (adjacency_matrix[i, j] = 3) <-> (j in tb[i])
);


% Constraints 5: Count the number of bonds for each atom (Corrected logic)
constraint forall(i in 1..natoms)(
  card(sb[i]) = sum(j in 1..nsingle_bonds)(bool2int(single_bond[i,j] != 0))
);

constraint forall(i in 1..natoms)(
  card(db[i]) = sum(j in 1..ndouble_bonds)(bool2int(double_bond[i,j] != 0))
);

constraint forall(i in 1..natoms)(
  card(tb[i]) = sum(j in 1..ntriple_bonds)(bool2int(triple_bond[i,j] != 0))
);


% Constraints 6: The graph is connected
constraint forall(i, j in 1..natoms)(
  exists(k in 1..natoms)(power_matrix(adjacency_matrix, k)[i,j] >= 1)
);

% Constraints 7: Rings have the correct size
constraint forall(i, j in 1..natoms)(
  adjacency_matrix[i, j] != 0
  /\ sum(row(ring_size, i)) > 0
  /\ sum(row(ring_size, j)) > 0 <-> (ring_matrix[i, j] + ring_matrix[j, i] > 0)
);

constraint forall(i, j in 1..natoms)(
  ring_matrix[i, j] + ring_matrix[j, i] < 2
);

constraint forall(i in 1..natoms)(
  if ring_size[i, 1] > 0 then
    exists(k in 3..4)(power_matrix(ring_matrix, k)[i,i] > 0)
  else
    forall(k in 3..4)(power_matrix(ring_matrix, k)[i,i] = 0)
  endif
);

constraint forall(i in 1..natoms)(
  if ring_size[i, 2] > 0 then
    exists(k in 5..7)(power_matrix(ring_matrix, k)[i,i] > 0)
  elseif ring_size[i, 1] = 0 then
    forall(k in 5..7)(power_matrix(ring_matrix, k)[i,i] = 0)
  endif
);

constraint forall(i in 1..natoms)(
  if ring_size[i, 3] > 0 then
    exists(k in 8..natoms)(power_matrix(ring_matrix, k)[i,i] > 0)
  elseif ring_size[i, 1] = 0 /\ ring_size[i, 2] = 0 then
    forall(k in 8..natoms)(power_matrix(ring_matrix, k)[i,i] = 0)
  endif
);

% Constraint 8: Symetry breaking constraint using lexicographical ordering
constraint forall(i, j in 1..natoms where i < j)(
  if are_identical(i, j) then
    lex_leq([adjacency_matrix[i, k] | k in 1..natoms], [adjacency_matrix[j, k] | k in 1..natoms])
  else
    true
  endif
);

solve satisfy;