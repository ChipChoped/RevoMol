function array[1..natoms, 1..natoms] of var 0..1: power_matrix(array[1..natoms, 1..natoms] of var int: M, int: p) =
  let {
    int: n = natoms;
    array[1..n, 1..n] of var 0..1: result;
  } in
  if p = 0 then
    array2d(1..n, 1..n, [if i = j then 1 else 0 endif | i, j in 1..n])
  elseif p = 1 then
    array2d(1..n, 1..n, [bool2int(M[i,j] != 0) | i,j in 1..n]) % Binarisation de M
  else
    let {
      array[1..n, 1..n] of var 0..1: temp = power_matrix(M, p div 2);
    } in
    if p mod 2 = 0 then
      matrix_mult(temp, temp)
    else
      matrix_mult(matrix_mult(temp, temp),  array2d(1..n, 1..n, [bool2int(M[i,j] != 0) | i,j in 1..n]))  % Binarisation de M
    endif
  endif;

  function array[1..natoms, 1..natoms] of var 0..1: matrix_mult(array[1..natoms, 1..natoms] of var 0..1: A, array[1..natoms, 1..natoms] of var 0..1: B) =
  let {
    int: n = natoms;
    array[1..n, 1..n] of var 0..1: result;
  } in
  array2d(1..n, 1..n, [bool2int(sum(k in 1..n)(A[i, k] * B[k, j]) != 0) | i, j in 1..n]); % Binarisation du résultat de la somme