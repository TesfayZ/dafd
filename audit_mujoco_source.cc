// Native source-build audit worker. Protocol matches benchmark_mujoco_integrator_order.py.
// Input on stdin: ntr nsteps h, qpos0[nq], initial velocities[ntr*nv].
#include <mujoco/mujoco.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 2 && argc != 3) return 2;
  char error[2048] = {};
  mjModel* m = mj_loadXML(argv[1], nullptr, error, sizeof(error));
  if (!m) { std::cerr << error; return 1; }
  mjData* d = mj_makeData(m);
  int ntr, steps; double h;
  std::cin >> ntr >> steps >> h;
  std::vector<mjtNum> q0(m->nq), v0(ntr*m->nv);
  for (auto& x : q0) std::cin >> x;
  for (auto& x : v0) std::cin >> x;
  if (!std::cin) return 3;
  for (int i=0; i<m->nv; i++) m->dof_damping[i] = m->dof_frictionloss[i] = 0;
  m->opt.viscosity = m->opt.density = 0;
  for (int i=0; i<m->njnt; i++) m->jnt_limited[i] = 0;
  m->opt.disableflags |= mjDSBL_CONTACT | mjDSBL_ACTUATION | mjDSBL_EQUALITY | mjDSBL_LIMIT;
  m->opt.enableflags |= mjENBL_ENERGY;
  auto restore = [&](const std::vector<mjtNum>& q, const std::vector<mjtNum>& v) {
    mj_resetData(m,d); mju_copy(d->qpos,q.data(),m->nq); mju_copy(d->qvel,v.data(),m->nv);
  };
  auto energy = [&](const std::vector<mjtNum>& q, const std::vector<mjtNum>& v) {
    restore(q,v); mj_forward(m,d); return d->energy[0]+d->energy[1];
  };
  std::cout << std::setprecision(17);
  if (argc == 3) {
    // Trajectory control: same total time at h, h/2, h/4, h/8;
    // reference h/256, Euclidean norm of the complete position vector.
    for (int t=0;t<ntr;t++) {
      std::vector<mjtNum> v(v0.begin()+t*m->nv,v0.begin()+(t+1)*m->nv), ref;
      for (int sub : {256,1,2,4,8}) {
        restore(q0,v); m->opt.timestep=h/sub;
        for (int k=0;k<steps*sub;k++) mj_step(m,d);
        for (int w : {mjWARN_BADQACC,mjWARN_BADQPOS,mjWARN_BADQVEL})
          if (d->warning[w].number) return 4;
        if (sub==256) ref.assign(d->qpos,d->qpos+m->nq);
        else {
          double err=0;
          for(int i=0;i<m->nq;i++) err+=(d->qpos[i]-ref[i])*(d->qpos[i]-ref[i]);
          std::cout << t << ',' << sub << ',' << std::sqrt(err) << '\n';
        }
      }
    }
    mj_deleteData(d);mj_deleteModel(m);return 0;
  }
  for (int t=0; t<ntr; t++) {
    auto q=q0;
    std::vector<mjtNum> v(v0.begin()+t*m->nv,v0.begin()+(t+1)*m->nv), nq, nv;
    double sums[4]={}, em=0;
    bool valid=true;
    for (int k=0;k<steps;k++) {
      double e0=energy(q,v); em+=std::abs(e0);
      for (int j=0;j<4;j++) {
        int sub=1<<j; restore(q,v); m->opt.timestep=h/sub;
        for (int s=0;s<sub;s++) mj_step(m,d);
        for (int w : {mjWARN_BADQACC,mjWARN_BADQPOS,mjWARN_BADQVEL})
          if (d->warning[w].number) valid=false;
        std::vector<mjtNum> aq(d->qpos,d->qpos+m->nq), av(d->qvel,d->qvel+m->nv);
        if (j==0) { nq=aq; nv=av; }
        sums[j]+=std::abs(energy(aq,av)-e0)/h;
      }
      q=nq;v=nv;
    }
    std::cout << t << ',' << em/steps;
    for (double x:sums) std::cout << ',' << (valid ? x/steps : NAN);
    std::cout << '\n';
  }
  mj_deleteData(d); mj_deleteModel(m);
}
