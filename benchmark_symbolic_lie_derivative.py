"""Exact Lie-derivative test on Acrobot in lifted coordinates (SymPy).  Paper: Section 3.1, Appendix A.
Output: Results/symbolic_lie_derivative_acrobot.csv

Coordinates (c1,s1,c2,s2,w1,w2); ideal I=<c1^2+s1^2-1, c2^2+s2^2-1>.
Vector field is rational (mass-matrix inverse); multiply by det to get a polynomial field
(positive time rescaling: preserves invariance of V and zero sets of Lie derivatives)."""
import sympy as sp
from results_io import save_csv
c1,s1,c2,s2,w1,w2,m,e = sp.symbols('c1 s1 c2 s2 w1 w2 m e')
G=sp.Rational(98,10); L1=1; LC=sp.Rational(1,2); I=1
def field(m2):
    phi1=sp.Rational(1,2)+m2*L1; phi2=m2*LC; hc=m2*L1*LC
    d1=sp.Rational(1,4)+m2*(1+sp.Rational(1,4)+c2)+2   # m2*(L1^2+LC^2+2 L1 LC c2)
    d2=m2*(sp.Rational(1,4)+LC*c2)+1
    d3=m2*sp.Rational(1,4)+1
    s12=s1*c2+c1*s2
    rhs1=-phi1*G*s1-phi2*G*s12+hc*s2*(2*w1*w2+w2**2)
    rhs2=-phi2*G*s12-hc*s2*w1**2
    det=sp.expand(d1*d3-d2**2)
    dd1=sp.expand(d3*rhs1-d2*rhs2); dd2=sp.expand(-d2*rhs1+d1*rhs2)   # = det * accel
    # d(c1)=-s1 w1, d(s1)=c1 w1, ...; multiplied by det
    f=[-s1*w1*det, c1*w1*det, -s2*w2*det, c2*w2*det, dd1, dd2]
    return f,det
vars_=[c1,s1,c2,s2,w1,w2]
def lie(p,f): return sp.expand(sum(sp.diff(p,v)*fi for v,fi in zip(vars_,f)))
E=(sp.Rational(7,4)*w1**2+sp.Rational(1,2)*c2*w1**2+sp.Rational(5,4)*w1*w2+sp.Rational(1,2)*c2*w1*w2
   +sp.Rational(5,8)*w2**2-sp.Rational(147,10)*c1-sp.Rational(49,10)*c1*c2+sp.Rational(49,10)*s1*s2)
gens=[c1**2+s1**2-1, c2**2+s2**2-1]
Gb=sp.groebner(gens,*vars_,e,order='lex',domain='QQ')
f0,det0=field(1)
print("det (m2=1) =",sp.factor(det0))
LE=lie(E,f0)
print("NF of det*L_f E at m2=1 :", Gb.reduce(LE)[1])
for g in gens: print("NF of L_f(g_i):", Gb.reduce(lie(g,f0))[1])
# fault signature: reference energy along faulted field
ff,detf=field(1+e)
Lf=sp.expand(lie(E,ff))
rem=Gb.reduce(Lf)[1]
rem=sp.expand(rem)
print("signature rho(e) = NF(det_e * L_{f_e} E_ref) has", len(sp.Poly(rem,*vars_).terms()),"terms; rho(0)=",rem.subs(e,0))
lin=sp.expand(sp.diff(rem,e).subs(e,0))
print("d rho/d e at e=0 has",len(sp.Poly(lin,*vars_).terms()),"terms")
# numeric value of g(x)=d/de [ L_{f_e}E ] at start state
import math
st={c1:math.cos(.1),s1:math.sin(.1),c2:math.cos(-.2),s2:math.sin(-.2),w1:.3,w2:-.15}
g_num=float(lin.subs(st))/float(det0.subs(st))
print("g(x0)=dD/de at start state (per unit e):",g_num)

save_csv("symbolic_lie_derivative_acrobot.csv", [
    dict(quantity="det M (m2=1)", value=str(sp.factor(det0))),
    dict(quantity="NF(det*L_f E) at m2=1", value=str(Gb.reduce(LE)[1])),
    dict(quantity="NF(L_f(c1^2+s1^2-1))", value=str(Gb.reduce(lie(gens[0], f0))[1])),
    dict(quantity="NF(L_f(c2^2+s2^2-1))", value=str(Gb.reduce(lie(gens[1], f0))[1])),
    dict(quantity="signature rho(0)", value=str(rem.subs(e, 0))),
    dict(quantity="d rho / d eps at 0 (terms)", value=str(len(sp.Poly(lin, *vars_).terms()))),
    dict(quantity="g(x0) = d D / d eps at start state", value=str(g_num)),
    dict(quantity="d rho / d eps at 0 (factored)", value=str(sp.factor(lin))),
])
