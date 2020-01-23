#include <scene.hpp>


Scene::Scene(size_t capacity) : capacity(capacity) {
    catchErr(cudaMalloc((void**)&this->volumes, capacity * sizeof(Volume *)));
}

__device__ glm::vec3 Scene::colorAt(const Ray& ray, curandState *randState) const {
    Ray r = ray;
    float energy = 1;
    for (size_t b = 0; b < 32; ++b) {
        //Find closest hit
        float closest = FLT_MAX;
        Hit hit;
        Material* mat;
        for (size_t i = 0; i < capacity; ++i) {
            Hit _hit;
            if (volumes[i]->intersect(r, 0.001f, closest, _hit)) {
                if (_hit.t < closest) {
                    hit = _hit;
                    closest = _hit.t;
                    mat = volumes[i]->getMat();
                }
            }
        }

        //Bounce if hit
        if (hit.t > 0) {
            // glm::vec3 bounceOut = reflect(glm::normalize(r.b), hit.normal);
            energy = mat->hit(hit, r, energy, randState);
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
