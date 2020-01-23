#include <scene.hpp>

#define RANDVEC3(R) glm::vec3(curand_uniform((R)),curand_uniform((R)),curand_uniform((R)))

__device__ glm::vec3 randVecUnitSphere(curandState *randState) {
    glm::vec3 point;
    do {
        point = 2.0f * RANDVEC3(randState) - glm::vec3(1, 1, 1);
    } while (glm::dot(point, point) >= 1.0f);
    return point;
}

Scene::Scene(size_t capacity) : capacity(capacity) {
    catchErr(cudaMalloc((void**)&this->volumes, capacity * sizeof(Volume *)));
}

__device__ glm::vec3 Scene::colorAt(const Ray& ray, curandState *randState) const {
    Hit hit;
    Ray r = ray;
    float energy = 1;
    for (size_t b = 0; b < 32; ++b) {
        //Find closest hit
        float minT = FLT_MAX;
        hit.t = 0;
        for (size_t i = 0; i < capacity; ++i) {
            Hit _hit;
            if (volumes[i]->intersect(r, _hit)) {
                if (_hit.t < minT) {
                    hit = _hit;
                    minT = _hit.t;
                }
            }
        }

        //Bounce if hit
        if (hit.t > 0) {
            glm::vec3 bounceOut = hit.normal + randVecUnitSphere(randState);
            energy *= 0.5f;
            r.a = hit.point;
            r.b = bounceOut;
        }
        
        //...or we return with sky color
        else {
            const glm::vec3 dir = glm::normalize(r.b);
            const float t = 0.5 * (dir.y + 1.0);
            const glm::vec3 sky = (1.0f - t) * glm::vec3(1, 1, 1) + t * glm::vec3(0.5, 0.7, 1);
            return energy * sky;
        }
    }

    //Energy lost
    return glm::vec3(0, 0, 0);
}
