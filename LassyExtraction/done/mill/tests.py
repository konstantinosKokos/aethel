from .types import Atom, Functor, Proof, Diamond

A, B, C, D = [Atom(i) for i in 'ABCD']
AB = Functor(A, B)
CD = Functor(Diamond('β', C), D)
AB_CD = Functor(AB, CD)


F = AB_CD.con('F')
f = AB.var(0)
g = CD.var(1)

aA = Diamond('α', A)
ax = aA.con('x')
x = Proof.undiamond('α', ax)
bx = Proof.diamond('β', x)


Ffx = Proof.apply(Proof.apply(F, f), bx)
assert type(Ffx) == D


