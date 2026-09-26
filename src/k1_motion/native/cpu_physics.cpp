// Batched native MuJoCo stepping. No Python callbacks, IPC, or changed physics.
#include <mujoco/mujoco.h>
#include <omp.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int J = 22, S = 17, A = 8, W = 179+A;
thread_local std::string error;
struct Batch {
  mjModel* model = nullptr;
  std::vector<mjData*> data;
  int workers, substeps, chunk, actual_workers = 0;
  double kp[J], kd[J], effort[J], speed[J], limits[2*J];
  double nominal[J], knee[J], operating[J];
  bool dynamic_actuator = false;
  bool operating_speed_guard = false;
  int sites[S];
  std::vector<double> effort_metric, saturation;
  std::vector<int> collision;
  std::vector<std::array<double,A>> safety;
  // Reused by the projected-command path; no GPU copy of joints already on host.
  std::vector<float> arm_state;
  std::vector<double> arm_commands;
  ~Batch() {
    for (auto* d : data) mj_deleteData(d);
    if (model) mj_deleteModel(model);
  }
  void pack(int i, float* output) const {
    const auto* d = data[i];
    auto* p = output + i*W;
    auto put = [&p](const mjtNum* v, int n) {
      for (int k=0; k<n; ++k) *p++ = static_cast<float>(v[k]);
    };
    put(d->qpos+7, J); put(d->qvel+6, J); put(d->qpos+3, 4);
    put(d->qvel+3, 3); put(d->qpos, 3); put(d->qvel, 3);
    for (int s : sites) put(d->site_xpos + 3*s, 3);
    for (int s : sites) put(d->xquat + 4*model->site_bodyid[s], 4);
    *p++ = static_cast<float>(effort_metric[i]);
    *p++ = static_cast<float>(saturation[i]);
    *p++ = static_cast<float>(collision[i]);
    for (double metric : safety[i]) *p++ = static_cast<float>(metric);
  }
};
template<class F> int checked(F f) {
  try { f(); error.clear(); return 0; }
  catch (const std::exception& e) { error=e.what(); return -1; }
}
struct ArmFeedback {
  mjModel* model = nullptr;
  std::vector<mjData*> data;
  int workers, left, right;
  double clearance;
  ~ArmFeedback() {
    for (auto* d : data) mj_deleteData(d);
    if (model) mj_deleteModel(model);
  }
};
}

extern "C" {
const char* k1_error() { return error.c_str(); }
int k1_mujoco_version() { return mjVERSION_HEADER; }
void* k1_arm_create(const mjModel* model, int workers, double clearance, const double* neutral) {
  try {
    if (!model || model->nq!=29 || model->nv!=28 || workers<1 || clearance<=0 || clearance>.03)
      throw std::runtime_error("Invalid arm feedback contract");
    auto b=std::make_unique<ArmFeedback>();
    b->model=mj_copyModel(nullptr, model); b->workers=workers; b->clearance=clearance;
    b->left=mj_name2id(model,mjOBJ_BODY,"left_elbow_yaw_link");
    b->right=mj_name2id(model,mjOBJ_BODY,"right_elbow_yaw_link");
    if (b->left<1 || b->right<1) throw std::runtime_error("Missing forearm bodies");
    std::fill(b->model->geom_margin,b->model->geom_margin+model->ngeom,.03);
    for (int i=0; i<workers; ++i) {
      auto* d=mj_makeData(b->model); b->data.push_back(d);
      std::copy(neutral,neutral+29,d->qpos);
      d->qpos[0]=0; d->qpos[1]=0; d->qpos[2]=1.5;
      d->qpos[3]=1; d->qpos[4]=0; d->qpos[5]=0; d->qpos[6]=0;
    }
    error.clear(); return b.release();
  } catch (const std::exception& e) { error=e.what(); return nullptr; }
}
int k1_arm_project(void* handle, int count, const float* q, const float* dq,
                   const float* target, float* output) {
  return checked([&] {
    auto& b=*static_cast<ArmFeedback*>(handle);
    for (int i=0; i<count*J; ++i)
      if (!std::isfinite(q[i]) || !std::isfinite(dq[i]) || !std::isfinite(target[i]))
        throw std::runtime_error("Nonfinite arm feedback input");
    #pragma omp parallel for num_threads(b.workers) schedule(static)
    for (int i=0; i<count; ++i) {
      auto* d=b.data[omp_get_thread_num()];
      for (int j=0; j<J; ++j) d->qpos[7+j]=q[i*J+j];
      // Projection only reads poses, contacts and Jacobians. Dynamics and the
      // constraint solver do not contribute to any of these inputs.
      mj_kinematics(b.model,d);
      mj_comPos(b.model,d);
      mj_collision(b.model,d);
      std::vector<std::array<double,J+1>> constraints;
      double j1[84], j2[84];
      for (int k=0; k<d->ncon; ++k) {
        const auto& c=d->contact[k];
        int b1=b.model->geom_bodyid[c.geom1], b2=b.model->geom_bodyid[c.geom2];
        if (!b1 || !b2 || c.dist>=.03) continue;
        if (b1!=b.left && b1!=b.right && b2!=b.left && b2!=b.right) continue;
        mj_jac(b.model,d,j1,nullptr,c.pos,b1); mj_jac(b.model,d,j2,nullptr,c.pos,b2);
        std::array<double,J+1> row{};
        double arm_norm=0, closing=0;
        for (int j=0; j<J; ++j) {
          for (int a=0; a<3; ++a) row[j]+=c.frame[a]*(j2[a*28+j+6]-j1[a*28+j+6]);
          if (j>=2 && j<10) arm_norm+=row[j]*row[j];
          closing+=row[j]*dq[i*J+j];
        }
        if (std::sqrt(arm_norm)<.03) continue;
        row[J]=b.clearance-c.dist-.04*std::min(0.,closing);
        constraints.push_back(row);
      }
      float* result=output+i*J;
      std::copy(target+i*J,target+(i+1)*J,result);
      for (int pass=0; pass<3; ++pass) for (const auto& row : constraints) {
        double missing=row[J], denominator=1e-8;
        for (int j=0; j<J; ++j) {
          missing-=row[j]*static_cast<float>(result[j]-q[i*J+j]);
          if (j>=2 && j<10) denominator+=row[j]*row[j];
        }
        if (missing>0) for (int j=2; j<10; ++j) {
          result[j]=static_cast<float>(result[j]+row[j]*missing/denominator);
          result[j]=std::clamp(result[j],target[i*J+j]-.15f,target[i*J+j]+.15f);
        }
      }
    }
  });
}
void k1_arm_destroy(void* handle) { delete static_cast<ArmFeedback*>(handle); }
void* k1_create(const mjModel* model, int count, int workers, int substeps, int chunk,
                const double* kp, const double* kd, const double* effort,
                const double* speed, const double* limits, const int* sites,
                const double* neutral, float* output, const double* actuator, int dynamic_actuator,
                int operating_speed_guard) {
  try {
    if (mj_version()!=mjVERSION_HEADER || !model || model->nq!=29 || model->nv!=28 || model->nu!=J)
      throw std::runtime_error("MuJoCo ABI or K1 layout mismatch");
    if (count<1 || workers<1 || substeps<1 || chunk<0)
      throw std::runtime_error("Invalid batch dimensions");
    auto b=std::make_unique<Batch>();
    b->workers=workers; b->substeps=substeps; b->chunk=chunk;
    b->model=mj_copyModel(nullptr, model);
    if (!b->model) throw std::runtime_error("Cannot copy model");
    std::copy(kp,kp+J,b->kp); std::copy(kd,kd+J,b->kd);
    std::copy(effort,effort+J,b->effort); std::copy(speed,speed+J,b->speed);
    std::copy(limits,limits+2*J,b->limits); std::copy(sites,sites+S,b->sites);
    std::copy(actuator,actuator+J,b->nominal);
    std::copy(actuator+J,actuator+2*J,b->knee);
    std::copy(actuator+2*J,actuator+3*J,b->operating);
    b->dynamic_actuator=dynamic_actuator!=0;
    b->operating_speed_guard=operating_speed_guard!=0;
    for (int s : b->sites) if (s<0 || s>=model->nsite) throw std::runtime_error("Invalid site");
    b->effort_metric.resize(count); b->saturation.resize(count); b->collision.resize(count);
    b->safety.resize(count);
    for (int i=0; i<count; ++i) {
      auto* d=mj_makeData(b->model);
      if (!d) throw std::runtime_error("Cannot allocate MuJoCo data");
      b->data.push_back(d);
    }
    #pragma omp parallel for num_threads(workers) schedule(static)
    for (int i=0; i<count; ++i) {
      if (i==0) b->actual_workers=omp_get_num_threads();
      auto* d=b->data[i];
      std::copy(neutral,neutral+29,d->qpos);
      mj_forward(b->model,d);
      b->pack(i,output);
    }
    error.clear();
    return b.release();
  } catch (const std::exception& e) { error=e.what(); return nullptr; }
}

int k1_reset(void* handle, const int* ids, int count, const double* states, float* output) {
  return checked([&] {
    auto& b=*static_cast<Batch*>(handle);
    // IDs are checked in Python as well, including uniqueness (parallel writes).
    for (int k=0; k<count; ++k)
      if (ids[k]<0 || ids[k]>=static_cast<int>(b.data.size())) throw std::runtime_error("Invalid reset ID");
    #pragma omp parallel for num_threads(b.workers) schedule(static) if(count>=b.workers)
    for (int k=0; k<count; ++k) {
      const int i=ids[k]; auto* d=b.data[i];
      mj_resetData(b.model,d);
      std::copy(states+k*57,states+k*57+29,d->qpos);
      std::copy(states+k*57+29,states+k*57+57,d->qvel);
      mj_forward(b.model,d);
      b.safety[i].fill(0.);
      b.pack(i,output);
    }
  });
}

int k1_step(void* handle, const double* targets, const double* velocities, float* output) {
  return checked([&] {
    auto& b=*static_cast<Batch*>(handle);
    const int count=b.data.size();
    std::atomic<bool> invalid{false};
    auto advance = [&](int i) {
      auto* d=b.data[i]; double q[J], v[J];
      for (int j=0; j<J; ++j) {
        q[j]=std::clamp(targets[i*J+j],b.limits[2*j],b.limits[2*j+1]);
        v[j]=std::clamp(velocities[i*J+j],-b.speed[j],b.speed[j]);
      }
      double effort=0, saturation=0; bool collision=false;
      auto& safety=b.safety[i]; safety.fill(0.);
      double requested[J], available[J];
      for (int step=0; step<b.substeps; ++step) {
        double e=0; int sat=0;
        for (int j=0; j<J; ++j) {
          const double torque=b.kp[j]*(q[j]-d->qpos[7+j])+b.kd[j]*(v[j]-d->qvel[6+j]);
          double bound=b.effort[j];
          if (b.dynamic_actuator) {
            const double denominator=std::max(b.nominal[j]-b.knee[j],1e-6);
            bound=std::clamp(b.effort[j]*(b.nominal[j]-std::abs(d->qvel[6+j]))/denominator,0.,b.effort[j]);
          }
          requested[j]=torque; available[j]=bound;
          sat += std::abs(torque)>=bound;
          d->ctrl[j]=std::clamp(torque,-bound,bound);
          if (b.operating_speed_guard && d->ctrl[j]*d->qvel[6+j]>0.) {
            const double factor=std::clamp((b.operating[j]-std::abs(d->qvel[6+j]))/(.1*b.operating[j]),0.,1.);
            d->ctrl[j]*=factor;
          }
          const double normalized=d->ctrl[j]/b.effort[j]; e += normalized*normalized;
        }
        effort += e/J; saturation += static_cast<double>(sat)/J;
        mj_step(b.model,d);
        // Actual post-integration state at every 2 ms, including the final tick.
        const double fraction=1./(J*b.substeps);
        for (int j=0; j<J; ++j) {
          const double operating=std::abs(d->qvel[6+j])/b.operating[j];
          const double nominal=std::abs(d->qvel[6+j])/b.nominal[j];
          const double joint_error=std::max({b.limits[2*j]-d->qpos[7+j],d->qpos[7+j]-b.limits[2*j+1],0.});
          safety[0]+=fraction*(operating>1.); safety[1]=std::max(safety[1],operating);
          safety[2]+=fraction*(nominal>1.); safety[3]=std::max(safety[3],nominal);
          safety[4]+=fraction*(joint_error>0.); safety[5]=std::max(safety[5],joint_error);
          safety[6]+=fraction*(std::abs(requested[j])>available[j]);
          const double excess=std::max(operating-1.,0.); safety[7]+=fraction*excess*excess;
        }
        for (int k=0; k<d->ncon; ++k) {
          const auto& c=d->contact[k];
          if (c.dist<=0 && b.model->geom_bodyid[c.geom1]!=0 && b.model->geom_bodyid[c.geom2]!=0)
            collision=true;
        }
      }
      for (int k=0; k<29; ++k) if (!std::isfinite(d->qpos[k])) invalid=true;
      for (int k=0; k<28; ++k) if (!std::isfinite(d->qvel[k])) invalid=true;
      for (int warning : {mjWARN_BADQPOS, mjWARN_BADQVEL, mjWARN_BADQACC, mjWARN_CONTACTFULL, mjWARN_CNSTRFULL})
        if (d->warning[warning].number) invalid=true;
      b.effort_metric[i]=effort/b.substeps; b.saturation[i]=saturation/b.substeps;
      // mj_step's derived transforms precede its final qpos integration. Refresh
      // only kinematics so reward landmarks/orientations share qpos's timestamp;
      // preserve the solved contacts, warm-start data and effort measurements.
      mj_kinematics(b.model,d);
      b.collision[i]=collision; b.pack(i,output);
    };
    if (b.chunk==0) {
      #pragma omp parallel for num_threads(b.workers) schedule(static)
      for (int i=0; i<count; ++i) advance(i);
    } else {
      #pragma omp parallel for num_threads(b.workers) schedule(dynamic,b.chunk)
      for (int i=0; i<count; ++i) advance(i);
    }
    if (invalid) throw std::runtime_error("Invalid MuJoCo state or solver capacity warning");
  });
}
int k1_foot_support(void* handle, float* output) {
  return checked([&] {
    const auto& b=*static_cast<Batch*>(handle);
    const int left=mj_name2id(b.model,mjOBJ_BODY,"left_ankle_roll_link");
    const int right=mj_name2id(b.model,mjOBJ_BODY,"right_ankle_roll_link");
    if (left<1 || right<1) throw std::runtime_error("Missing foot bodies");
    const int count=b.data.size();
    #pragma omp parallel for num_threads(b.workers) schedule(static)
    for (int i=0; i<count; ++i) {
      const auto* d=b.data[i];
      double forces[2]={}, speeds[2]={};
      for (int k=0; k<d->ncon; ++k) {
        const auto& c=d->contact[k];
        if (c.efc_address<0) continue;
        const int first=b.model->geom_bodyid[c.geom1], second=b.model->geom_bodyid[c.geom2];
        const int body=first==0 ? second : second==0 ? first : -1;
        const int foot=body==left ? 0 : body==right ? 1 : -1;
        if (foot<0) continue;
        mjtNum force[6], jacobian[3*28], velocity[3];
        mj_contactForce(b.model,d,k,force);
        const double normal=std::max(0.,force[0]);
        if (normal==0) continue;
        mj_jac(b.model,d,jacobian,nullptr,c.pos,body);
        mju_mulMatVec(velocity,jacobian,d->qvel,3,28);
        const double along=mju_dot3(velocity,c.frame);
        double tangential=0;
        for (int axis=0; axis<3; ++axis) {
          const double v=velocity[axis]-along*c.frame[axis]; tangential+=v*v;
        }
        forces[foot]+=normal; speeds[foot]+=normal*tangential;
      }
      for (int foot=0; foot<2; ++foot) {
        output[i*6+foot]=forces[foot];
        output[i*6+2+foot]=std::sqrt(speeds[foot]/std::max(forces[foot],1e-12));
        output[i*6+4+foot]=0;
      }
    }
  });
}

int k1_step_projected(void* handle, void* arm_handle, const float* commands,
                     const float* limits, const float* cap, float* projected,
                     float* output, double* timing) {
  return checked([&] {
    auto& b=*static_cast<Batch*>(handle);
    auto& arm=*static_cast<ArmFeedback*>(arm_handle);
    const int count=b.data.size(), size=count*J;
    if (arm.workers!=b.workers) throw std::runtime_error("Arm/physics worker mismatch");
    b.arm_state.resize(2*size);
    b.arm_commands.resize(2*size);
    for (int i=0; i<count; ++i) {
      // Match the float32 state previously published to the policy device.
      std::copy(output+i*W,output+i*W+J,b.arm_state.data()+i*J);
      std::copy(output+i*W+J,output+i*W+2*J,b.arm_state.data()+size+i*J);
    }
    const double before=omp_get_wtime();
    if (k1_arm_project(arm_handle,count,b.arm_state.data(),b.arm_state.data()+size,
                       commands,projected)) throw std::runtime_error(error);
    timing[0]=omp_get_wtime()-before;
    for (int i=0; i<size; ++i) {
      const int j=i%J;
      // Preserve both GPU-side clamps and float32 rounding before the float64
      // servo input conversion. Never bypass joint or command-rate limits.
      const float bounded=std::clamp(projected[i],limits[2*j],limits[2*j+1]);
      const float lower=commands[2*size+i]-cap[j], upper=commands[2*size+i]+cap[j];
      projected[i]=std::clamp(bounded,lower,upper);
      b.arm_commands[i]=projected[i];
      b.arm_commands[size+i]=commands[size+i];
    }
    const double stepping=omp_get_wtime();
    if (k1_step(handle,b.arm_commands.data(),b.arm_commands.data()+size,output))
      throw std::runtime_error(error);
    timing[1]=omp_get_wtime()-stepping;
  });
}
void k1_destroy(void* handle) { delete static_cast<Batch*>(handle); }
int k1_workers(void* handle) { return static_cast<Batch*>(handle)->actual_workers; }
}
