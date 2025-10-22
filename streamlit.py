import streamlit as st
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt

# Set up the device for PyTorch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. CORE COMPONENTS FROM NOTEBOOK ---

# Named tuple for Replay Buffer experience
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.capacity = capacity

    def push(self, *args):
        """Save an experience."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Randomly sample a batch of experiences."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class QNetwork(nn.Module):
    """Deep Q-Network (DQN) architecture."""
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # Simple two-layer neural network
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """The DQN agent that interacts with and learns from the environment."""
    def __init__(self, state_size, action_size, gamma, lr, memory_size, batch_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size

        # Q-Networks (Policy and Target)
        self.policy_net = QNetwork(state_size, action_size).to(DEVICE)
        self.target_net = QNetwork(state_size, action_size).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval() # Target network is only used for prediction

        # Optimizer and Replay Buffer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(memory_size)
        self.criterion = nn.MSELoss()

    def select_action(self, state, epsilon):
        """Epsilon-greedy action selection."""
        # Convert state index (int) to one-hot vector (tensor)
        state_tensor = torch.zeros(self.state_size, device=DEVICE)
        state_tensor[state] = 1.0
        state_tensor = state_tensor.unsqueeze(0)

        if random.random() > epsilon:
            with torch.no_grad():
                # Policy net predicts the best action
                action_values = self.policy_net(state_tensor)
            return action_values.max(1)[1].view(1, 1).item()
        else:
            return random.randrange(self.action_size)

    def learn(self):
        """Update the policy network using a batch of experience."""
        if len(self.memory) < self.batch_size:
            return # Not enough experience yet

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # Convert state indices to one-hot tensors for batch processing
        def to_one_hot_batch(indices):
            indices_list = [i.item() if isinstance(i, torch.Tensor) else i for i in indices]
            tensor = torch.zeros(len(indices_list), self.state_size, device=DEVICE)
            tensor[torch.arange(len(indices_list)), indices_list] = 1.0
            return tensor

        state_batch = to_one_hot_batch(batch.state)
        next_state_batch = to_one_hot_batch(batch.next_state)

        action_batch = torch.LongTensor(batch.action).to(DEVICE).unsqueeze(1)
        reward_batch = torch.FloatTensor(batch.reward).to(DEVICE)
        done_batch = torch.BoolTensor(batch.done).to(DEVICE)

        # Compute Q(s_t, a) - the Q-value for the action taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)

        # Compute V(s_{t+1}) = max_a Q_target(s_{t+1}, a) for non-terminal next states
        next_state_values = torch.zeros(self.batch_size, device=DEVICE)
        non_final_mask = ~done_batch
        
        # Select max Q value from target network for non-terminal states
        next_state_values[non_final_mask] = self.target_net(next_state_batch[non_final_mask]).max(1)[0].detach()

        # Compute the expected Q values: R_t + gamma * V(s_{t+1})
        expected_state_action_values = reward_batch + (self.gamma * next_state_values)

        # Compute loss (MSE between predicted Q and expected Q)
        loss = self.criterion(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

# --- 2. TRAINING AND UTILITY FUNCTIONS ---

@st.cache_resource(show_spinner=False)
def create_env(map_name, is_slippery):
    """Initialize the FrozenLake environment."""
    env_id = 'FrozenLake-v1'
    
    # 1. Create the environment (which includes the TimeLimit wrapper)
    if map_name == '4x4':
        desc = ["SFFF", "FHFH", "FFFH", "HFFG"]
        env = gym.make(env_id, is_slippery=is_slippery, desc=desc)
    elif map_name == '8x8':
        env = gym.make(env_id, is_slippery=is_slippery, map_name='8x8')
        # Access the unwrapped environment to get the description
        desc = env.unwrapped.desc.tolist() 
    else:
        env = gym.make(env_id, is_slippery=is_slippery)
        desc = env.unwrapped.desc.tolist()

    # 2. Store the map description to session state
    # This prevents accessing env.desc (which is missing on the TimeLimit wrapper) later.
    st.session_state.map_description = desc
    
    state_size = env.observation_space.n
    action_size = env.action_space.n
    return env, state_size, action_size

def run_training(env, agent, num_episodes, target_update, eps_start, eps_end, eps_decay):
    """Main training loop adapted for Streamlit."""
    episode_rewards = []
    
    # Placeholders for dynamic updates
    status_text = st.empty()
    chart_placeholder = st.empty()
    
    total_steps = 0
    
    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        
        # Epsilon calculation (exponential decay)
        epsilon = eps_end + (eps_start - eps_end) * np.exp(-1. * total_steps / eps_decay)
        
        while not done and not truncated:
            action = agent.select_action(state, epsilon)
            
            # Perform action
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store transition in replay buffer
            agent.memory.push(torch.tensor([state]), torch.tensor([action]), 
                              torch.tensor([reward]), torch.tensor([next_state]), 
                              torch.tensor([done or truncated]))

            # Move to the next state
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            # Perform one optimization step
            agent.learn()
            
        episode_rewards.append(episode_reward)

        # Update target network
        if episode % target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
        # Update Streamlit UI every 50 episodes
        if episode % 50 == 0 or episode == num_episodes:
            status_text.info(f"Episode: {episode}/{num_episodes} | Avg Reward (last 50): {np.mean(episode_rewards[-50:]):.4f} | Epsilon: {epsilon:.4f}")
            
            # Update chart
            chart_data = {'Episode': list(range(1, len(episode_rewards) + 1)), 'Reward': episode_rewards}
            chart_placeholder.line_chart(chart_data, x='Episode', y='Reward', height=300)

    # After training, finalize the status and chart
    status_text.success(f"Training Complete! Total Episodes: {num_episodes}")
    
    return episode_rewards

def extract_policy(env, agent):
    """Extract the optimal deterministic policy from the learned Q-network."""
    policy = np.zeros(env.observation_space.n, dtype=int)
    
    for state in range(env.observation_space.n):
        # Convert state index (int) to one-hot vector (tensor)
        state_tensor = torch.zeros(env.observation_space.n, device=DEVICE)
        state_tensor[state] = 1.0
        state_tensor = state_tensor.unsqueeze(0)
        
        with torch.no_grad():
            q_values = agent.policy_net(state_tensor)
        
        policy[state] = q_values.argmax().item()
    
    return policy

def visualize_policy(env, policy):
    """Renders the policy grid using Matplotlib."""
    # Rely exclusively on the map description stored in session state
    map_desc = st.session_state.get('map_description')
    
    if map_desc is None:
        # Fallback for safety, though create_env should always set this now
        map_desc = env.unwrapped.desc.tolist() 

    # Convert state indices to a grid format (assuming square map)
    map_size = int(np.sqrt(env.observation_space.n))
    
    # Action mapping: 0: Left, 1: Down, 2: Right, 3: Up
    action_symbols = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    
    grid = policy.reshape((map_size, map_size))
    
    fig, ax = plt.subplots(figsize=(map_size, map_size))
    ax.set_title("Learned Optimal Policy (Action per State)")
    
    # Draw the grid
    ax.set_xticks(np.arange(map_size + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(map_size + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks([])
    ax.set_yticks([])

    # Display state type (S, F, H, G) and action symbol
    for i in range(map_size):
        for j in range(map_size):
            # i is row (y), j is column (x)
            state_idx = i * map_size + j
            action = grid[i, j]
            symbol = action_symbols.get(action, '?')
            
            # Get the state type from the map description
            state_element = map_desc[i][j]
            
            # Ensure the state character is a string (handle potential bytes from np array conversion)
            if isinstance(state_element, (bytes, np.bytes_)):
                state_char = state_element.decode('utf-8')
            else:
                # If it's already a string, take the character
                state_char = str(state_element)
            
            # Color map for the grid cells
            color = 'lightgreen' if state_char == 'G' else ('lightcoral' if state_char == 'H' else ('skyblue' if state_char == 'S' else 'white'))
            
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor='black', linewidth=1))

            # Center text on cell
            ax.text(j, i, f"{state_char}\n{symbol}", 
                    ha='center', va='center', fontsize=16, 
                    weight='bold', color='black')

    # Flip y-axis to match typical grid representation (0,0 top-left)
    ax.invert_yaxis()
    
    st.pyplot(fig)


# --- 3. STREAMLIT APPLICATION LAYOUT ---

st.set_page_config(layout="wide", page_title="DQN Agent for FrozenLake")

st.title("❄️ Deep Q-Learning for FrozenLake")
st.markdown("Use the sidebar to configure the environment and hyperparameters, then click **Train Agent** to see the DQN in action.")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("1. Environment Setup")
    
    map_choice = st.selectbox(
        "Select Map Size",
        ('4x4', '8x8'),
        key='map_choice'
    )
    
    is_slippery = st.checkbox(
        "Slippery Environment (Stochastic)",
        value=True,
        key='is_slippery',
        help="If checked, the agent moves in the intended direction only 1/3 of the time. If unchecked, movement is deterministic."
    )
    
    st.header("2. Hyperparameters")
    
    num_episodes = st.number_input("Total Episodes", min_value=100, max_value=20000, value=5000, step=100)
    gamma = st.slider("Discount Factor (Gamma)", 0.0, 1.0, 0.99, 0.01)
    lr = st.number_input("Learning Rate", min_value=1e-5, max_value=1e-2, value=1e-3, format="%.5f")
    
    st.subheader("Epsilon Schedule")
    eps_start = st.slider("Epsilon Start", 0.0, 1.0, 1.0, 0.05)
    eps_end = st.slider("Epsilon End", 0.0, 1.0, 0.05, 0.05)
    eps_decay = st.number_input("Epsilon Decay Steps", min_value=100, value=2000, step=100)
    
    st.subheader("DQN Params")
    batch_size = st.number_input("Batch Size", min_value=16, max_value=128, value=64, step=16)
    memory_size = st.number_input("Replay Memory Size", min_value=1000, max_value=50000, value=10000, step=1000)
    target_update = st.number_input("Target Network Update Frequency", min_value=1, value=10, step=1)
    
    # Initialize the environment and agent based on current settings
    env, state_size, action_size = create_env(map_choice, is_slippery)
    
    if 'agent' not in st.session_state or st.button("Reset Agent/Env"):
        # This block now relies on create_env also setting st.session_state.map_description
        st.session_state.env = env
        st.session_state.state_size = state_size
        st.session_state.action_size = action_size
        st.session_state.agent = DQNAgent(state_size, action_size, gamma, lr, memory_size, batch_size)
        st.session_state.episode_rewards = []
        st.session_state.policy = None
        st.session_state.config_hash = hash((map_choice, is_slippery, gamma, lr, num_episodes, eps_start, eps_end, eps_decay, batch_size, memory_size, target_update))
        st.success("Agent and Environment Initialized.")

# --- Main Content Area ---

if st.sidebar.button("🚀 Train Agent", type="primary", use_container_width=True):
    # Rerun the initialization if configuration has changed since last training
    current_hash = hash((map_choice, is_slippery, gamma, lr, num_episodes, eps_start, eps_end, eps_decay, batch_size, memory_size, target_update))
    if current_hash != st.session_state.config_hash:
        st.warning("Configuration changed. Re-initializing Agent...")
        # Re-creating environment also updates st.session_state.map_description
        st.session_state.env, st.session_state.state_size, st.session_state.action_size = create_env(map_choice, is_slippery)
        st.session_state.agent = DQNAgent(
            st.session_state.state_size, 
            st.session_state.action_size, 
            gamma, lr, memory_size, batch_size
        )
        st.session_state.episode_rewards = []
        st.session_state.policy = None
        st.session_state.config_hash = current_hash
        st.success("Agent and Environment Re-initialized.")


    st.subheader("Training Progress: Reward per Episode")
    
    # Run the training process with a spinner
    with st.spinner(f"Training DQN for {num_episodes} episodes..."):
        rewards = run_training(
            st.session_state.env,
            st.session_state.agent,
            num_episodes,
            target_update,
            eps_start,
            eps_end,
            eps_decay
        )
    
    st.session_state.episode_rewards = rewards
    
    st.subheader("Optimal Policy Extraction")
    st.session_state.policy = extract_policy(st.session_state.env, st.session_state.agent)
    st.success("Policy extracted successfully!")

# --- Display Results (if available) ---

if st.session_state.get('episode_rewards'):
    st.header("Final Policy Visualization")
    visualize_policy(st.session_state.env, st.session_state.policy)

    st.header("Training Statistics")
    avg_reward = np.mean(st.session_state.episode_rewards[-100:])
    st.metric(label="Average Reward (Last 100 Episodes)", value=f"{avg_reward:.4f}")
    
    # Final plot confirmation
    st.subheader("Reward History")
    chart_data = {'Episode': list(range(1, len(st.session_state.episode_rewards) + 1)), 'Reward': st.session_state.episode_rewards}
    st.line_chart(chart_data, x='Episode', y='Reward', height=300)

elif st.session_state.get('policy'):
    # If the app was rerun but training wasn't, show the previous results
    st.header("Current Learned Policy")
    visualize_policy(st.session_state.env, st.session_state.policy)
    st.info("Click 'Train Agent' in the sidebar to re-run the training process with new parameters.")
else:
    st.info("Configure your environment and hyperparameters in the sidebar, and click '🚀 Train Agent' to begin the Deep Q-Learning process.")
