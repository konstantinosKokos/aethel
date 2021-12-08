from .types import Atom, Functor, Proof

A, B, C, D = [Atom(i) for i in 'ABCD']
AB = Functor(A, B)
CD = Functor(C, D)
AB_CD = Functor(AB, CD)


F = AB_CD.con('F')
f = AB.var(0)
g = CD.var(1)
x = A.con('x')

Ffx = Proof.apply(Proof.apply(F, f), x)
assert type(Ffx) == D