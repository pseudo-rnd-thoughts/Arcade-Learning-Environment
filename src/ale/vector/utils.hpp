#ifndef ALE_VECTOR_UTILS_HPP_
#define ALE_VECTOR_UTILS_HPP_

#include <vector>
#include <atomic>
#include <mutex>
#include <memory>

#ifndef MOODYCAMEL_DELETE_FUNCTION
    #define MOODYCAMEL_DELETE_FUNCTION = delete
#endif

#include "ale/common/Constants.h"
#include "ale/external/lightweightsemaphore.h"

namespace ale::vector {

    /**
     * ActionSlice represents a single action or command to be processed by a worker thread
     */
    struct ActionSlice {
        int env_id;        // ID of the environment to apply the action to
        bool force_reset;  // Whether to force a reset of the environment
    };

    /**
     * EnvironmentAction represents an action to be taken in an environment
     */
    struct EnvironmentAction {
        int env_id;            // ID of the environment to apply the action to
        int action_id;         // ID of the action to take
        float paddle_strength; // Strength for paddle-based games (default: 1.0)
    };

    /**
     * BatchData represents collected data for a batch of timesteps
     * Used to return data from StateBuffer::collect()
     */
    struct BatchData {
        std::vector<int> env_ids;
        std::vector<uint8_t> observations;      // Flat array [batch_size * obs_size]
        std::vector<reward_t> rewards;
        std::vector<uint8_t> terminated;        // Using uint8_t instead of bool for .data() access
        std::vector<uint8_t> truncated;         // Using uint8_t instead of bool for .data() access
        std::vector<int> lives;
        std::vector<int> frame_numbers;
        std::vector<int> episode_frame_numbers;

        // For SameStep autoreset mode
        std::vector<uint8_t> final_observations; // Flat array [batch_size * obs_size]
        std::vector<uint8_t> has_final_observation; // Using uint8_t instead of bool for .data() access

        size_t batch_size;
        size_t obs_size;
    };

    /**
     * Observation format enumeration
     */
    enum class ObsFormat {
        Grayscale,  // Single channel grayscale observations
        RGB         // Three channel RGB observations
    };

    enum class AutoresetMode {
        NextStep,  // Will reset the sub-environment in the next step if the episode ended in the previous timestep
        SameStep   // Will reset the sub-environment in the same timestep if the episode ended
    };

    /**
     * Lock-free queue for actions to be processed by worker threads
     */
    class ActionQueue {
    public:
        explicit ActionQueue(const std::size_t num_envs)
            : alloc_ptr_(0),
              done_ptr_(0),
              queue_size_(num_envs * 2),
              queue_(queue_size_),
              sem_(0),
              sem_enqueue_(1),
              sem_dequeue_(1) {}

        /**
         * Enqueue multiple actions at once
         */
        void enqueue_bulk(const std::vector<ActionSlice>& actions) {
            while (!sem_enqueue_.wait()) {}

            const uint64_t pos = alloc_ptr_.fetch_add(actions.size());
            for (std::size_t i = 0; i < actions.size(); ++i) {
                queue_[(pos + i) % queue_size_] = actions[i];
            }

            sem_.signal(actions.size());
            sem_enqueue_.signal(1);
        }

        /**
         * Dequeue a single action
         */
        ActionSlice dequeue() {
            while (!sem_.wait()) {}
            while (!sem_dequeue_.wait()) {}

            const auto ptr = done_ptr_.fetch_add(1);
            const auto ret = queue_[ptr % queue_size_];

            sem_dequeue_.signal(1);
            return ret;
        }

        /**
         * Get the approximate size of the queue
         */
        std::size_t size_approx() const {
            return alloc_ptr_ - done_ptr_;
        }

    private:
        std::atomic<uint64_t> alloc_ptr_;  // Pointer to next allocation position
        std::atomic<uint64_t> done_ptr_;   // Pointer to next dequeue position
        std::size_t queue_size_;           // Size of the queue
        std::vector<ActionSlice> queue_;   // The actual queue data
        moodycamel::LightweightSemaphore sem_;           // Semaphore for queue access
        moodycamel::LightweightSemaphore sem_enqueue_;   // Semaphore for enqueue operations
        moodycamel::LightweightSemaphore sem_dequeue_;   // Semaphore for dequeue operations
    };

    /**
     * StateBuffer handles the collection of timesteps from environments using zero-copy architecture
     *
     * Two modes of operation:
     * 1. Ordered mode (batch_size == num_envs): Waits for all env_ids to be filled
     * 2. Unordered mode (batch_size != num_envs): Uses circular buffer for continuous operation
     */
    class StateBuffer {
    public:
        StateBuffer(const std::size_t batch_size, const std::size_t num_envs, const std::size_t obs_size)
            : batch_size_(batch_size),
              num_envs_(num_envs),
              obs_size_(obs_size),
              ordered_mode_(batch_size == num_envs),
              // Pre-allocate all buffers (Structure-of-Arrays)
              observations_(num_envs * obs_size),
              env_ids_(num_envs),
              rewards_(num_envs),
              terminated_(num_envs),
              truncated_(num_envs),
              lives_(num_envs),
              frame_numbers_(num_envs),
              episode_frame_numbers_(num_envs),
              // For SameStep autoreset
              final_observations_(num_envs * obs_size),
              has_final_observation_(num_envs, 0),
              // Atomic counters
              count_(0),
              write_idx_(0),
              read_idx_(0),
              sem_ready_(0),      // Initially no batches ready
              sem_read_(1) {      // Allow one reader at a time
        }

        /**
         * Get pointer to observation buffer for writing
         * @param slot_idx The slot index in the buffer (NOT env_id in unordered mode)
         * @return Pointer to the observation buffer for this slot
         */
        uint8_t* get_observation_buffer(std::size_t slot_idx) {
            return observations_.data() + slot_idx * obs_size_;
        }

        /**
         * Get pointer to final observation buffer for writing (SameStep autoreset)
         * @param env_id The environment ID (always indexed by env_id)
         * @return Pointer to the final observation buffer for this environment
         */
        uint8_t* get_final_observation_buffer(int env_id) {
            return final_observations_.data() + env_id * obs_size_;
        }

        /**
         * Write metadata for a timestep
         * Multiple threads can write simultaneously
         * @param slot_idx The slot index determined by allocation
         * @param env_id The environment ID
         */
        void write_metadata(std::size_t slot_idx, int env_id, reward_t reward,
                          bool terminated, bool truncated, int lives,
                          int frame_number, int episode_frame_number,
                          bool has_final_obs = false) {
            env_ids_[slot_idx] = env_id;
            rewards_[slot_idx] = reward;
            terminated_[slot_idx] = terminated;
            truncated_[slot_idx] = truncated;
            lives_[slot_idx] = lives;
            frame_numbers_[slot_idx] = frame_number;
            episode_frame_numbers_[slot_idx] = episode_frame_number;
            has_final_observation_[slot_idx] = has_final_obs;

            // Atomically increment count and signal if batch is ready
            if (ordered_mode_) {
                const auto old_count = count_.fetch_add(1);
                if (old_count + 1 == batch_size_) {
                    sem_ready_.signal(1);
                }
            } else {
                const auto old_count = count_.fetch_add(1);
                // Signal if we just crossed a batch boundary
                if ((old_count + 1) / batch_size_ > old_count / batch_size_) {
                    sem_ready_.signal(1);
                }
            }
        }

        /**
         * Allocate a slot index for writing in unordered mode
         * @return The allocated slot index
         */
        std::size_t allocate_slot() {
            return write_idx_.fetch_add(1) % num_envs_;
        }

        /**
         * Collect timesteps when ready and return them
         *
         * @return BatchData containing the collected timesteps
         */
        BatchData collect() {
            // Wait until a batch is ready
            while (!sem_ready_.wait()) {}

            // Acquire read semaphore
            while (!sem_read_.wait()) {}

            // Prepare result
            BatchData result;
            result.batch_size = batch_size_;
            result.obs_size = obs_size_;

            // Pre-allocate result vectors
            result.env_ids.resize(batch_size_);
            result.observations.resize(batch_size_ * obs_size_);
            result.rewards.resize(batch_size_);
            result.terminated.resize(batch_size_);
            result.truncated.resize(batch_size_);
            result.lives.resize(batch_size_);
            result.frame_numbers.resize(batch_size_);
            result.episode_frame_numbers.resize(batch_size_);
            result.has_final_observation.resize(batch_size_);

            if (ordered_mode_) {
                // In ordered mode, read in env_id order (contiguous copy)
                std::memcpy(result.env_ids.data(), env_ids_.data(), batch_size_ * sizeof(int));
                std::memcpy(result.observations.data(), observations_.data(), batch_size_ * obs_size_);
                std::memcpy(result.rewards.data(), rewards_.data(), batch_size_ * sizeof(reward_t));
                std::memcpy(result.terminated.data(), terminated_.data(), batch_size_ * sizeof(uint8_t));
                std::memcpy(result.truncated.data(), truncated_.data(), batch_size_ * sizeof(uint8_t));
                std::memcpy(result.lives.data(), lives_.data(), batch_size_ * sizeof(int));
                std::memcpy(result.frame_numbers.data(), frame_numbers_.data(), batch_size_ * sizeof(int));
                std::memcpy(result.episode_frame_numbers.data(), episode_frame_numbers_.data(), batch_size_ * sizeof(int));
                std::memcpy(result.has_final_observation.data(), has_final_observation_.data(), batch_size_ * sizeof(uint8_t));

                // Reset count for ordered mode
                count_.store(0);
            } else {
                // In unordered mode, read from circular buffer
                for (size_t i = 0; i < batch_size_; ++i) {
                    const auto idx = read_idx_.fetch_add(1) % num_envs_;

                    result.env_ids[i] = env_ids_[idx];
                    std::memcpy(result.observations.data() + i * obs_size_,
                              observations_.data() + idx * obs_size_,
                              obs_size_);
                    result.rewards[i] = rewards_[idx];
                    result.terminated[i] = terminated_[idx];
                    result.truncated[i] = truncated_[idx];
                    result.lives[i] = lives_[idx];
                    result.frame_numbers[i] = frame_numbers_[idx];
                    result.episode_frame_numbers[i] = episode_frame_numbers_[idx];
                    result.has_final_observation[i] = has_final_observation_[idx];
                }

                // Atomically decrease count by batch_size_
                count_.fetch_sub(batch_size_);
            }

            // Copy final observations if needed (indexed by env_id, not slot)
            bool any_final_obs = false;
            for (size_t i = 0; i < batch_size_; ++i) {
                if (result.has_final_observation[i]) {
                    any_final_obs = true;
                    break;
                }
            }

            if (any_final_obs) {
                result.final_observations.resize(batch_size_ * obs_size_);
                for (size_t i = 0; i < batch_size_; ++i) {
                    if (result.has_final_observation[i]) {
                        int env_id = result.env_ids[i];
                        std::memcpy(result.final_observations.data() + i * obs_size_,
                                  final_observations_.data() + env_id * obs_size_,
                                  obs_size_);
                    } else {
                        // Copy current observation as placeholder
                        std::memcpy(result.final_observations.data() + i * obs_size_,
                                  result.observations.data() + i * obs_size_,
                                  obs_size_);
                    }
                }
            }

            // Release read semaphore
            sem_read_.signal(1);

            return result;
        }

        /**
         * Get the number of timesteps currently buffered
         */
        size_t filled_timesteps() const {
            return count_.load();
        }

    private:
        const std::size_t batch_size_;                    // Size of each batch
        const std::size_t num_envs_;                      // Number of environments
        const std::size_t obs_size_;                      // Observation size per environment
        const bool ordered_mode_;                         // Whether we're in ordered mode

        // Structure-of-Arrays storage (pre-allocated)
        std::vector<uint8_t> observations_;               // [num_envs * obs_size]
        std::vector<int> env_ids_;                        // [num_envs]
        std::vector<reward_t> rewards_;                   // [num_envs]
        std::vector<uint8_t> terminated_;                 // [num_envs] - Using uint8_t instead of bool
        std::vector<uint8_t> truncated_;                  // [num_envs] - Using uint8_t instead of bool
        std::vector<int> lives_;                          // [num_envs]
        std::vector<int> frame_numbers_;                  // [num_envs]
        std::vector<int> episode_frame_numbers_;          // [num_envs]

        // For SameStep autoreset (indexed by env_id, not slot)
        std::vector<uint8_t> final_observations_;         // [num_envs * obs_size]
        std::vector<uint8_t> has_final_observation_;      // [num_envs] - Using uint8_t instead of bool

        // Atomic counters for lock-free operations
        std::atomic<std::size_t> count_;                  // Current count of available timesteps
        std::atomic<std::size_t> write_idx_;              // Write position (for unordered mode)
        std::atomic<std::size_t> read_idx_;               // Read position (for unordered mode)

        // Semaphores for coordination
        moodycamel::LightweightSemaphore sem_ready_;      // Signals when a batch is ready for collection
        moodycamel::LightweightSemaphore sem_read_;       // Controls access to read operations
    };
}

#endif // ALE_VECTOR_UTILS_HPP_
