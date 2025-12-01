import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from jax import value_and_grad, jit
import optax
import os

# --- 1. 物理エンジン定義 (Standalone版) ---
def create_engine():
    def directional_coupler():
        val = 1.0 / jnp.sqrt(2.0)
        return jnp.array([[val, val * 1j], [val * 1j, val]])

    def pockels_phase_shifter(voltage):
        L = 2000e-6; d = 0.3e-6; wl = 1.55e-6; n = 3.5; r = 100e-12
        E = voltage / d
        dn = 0.5 * (n**3) * r * E
        phi = (2 * jnp.pi / wl) * dn * L
        return jnp.array([[jnp.exp(1j * phi), 0], [0, 1.0 + 0j]])

    def mzi_switch(voltage):
        DC = directional_coupler()
        PS = pockels_phase_shifter(voltage)
        return jnp.dot(DC, jnp.dot(PS, DC))

    @jit
    def simulate_mesh(voltages):
        T0 = mzi_switch(voltages[0]); T1 = mzi_switch(voltages[1])
        L1 = jnp.block([[T0, jnp.zeros((2,2))], [jnp.zeros((2,2)), T1]])
        T2 = mzi_switch(voltages[2])
        L2 = jnp.eye(4, dtype=complex); L2 = L2.at[1:3, 1:3].set(T2)
        T3 = mzi_switch(voltages[3]); T4 = mzi_switch(voltages[4])
        L3 = jnp.block([[T3, jnp.zeros((2,2))], [jnp.zeros((2,2)), T4]])
        T5 = mzi_switch(voltages[5])
        L4 = jnp.eye(4, dtype=complex); L4 = L4.at[1:3, 1:3].set(T5)
        U = jnp.dot(L4, jnp.dot(L3, jnp.dot(L2, L1)))
        return U
    return simulate_mesh

def run_quantization_study():
    print("🚀 DiffPhoton: DAC Quantization Study (Bit-Depth Analysis)...")
    print("   Goal: Determine the minimum bits required for the driver circuit.")
    
    # ターゲット: 画像分類 (0 vs 1, 真・直交エンコーディング)
    img_1 = jnp.array([0.0, 0.70710678, 0.0, 0.70710678]) + 0j
    target_1 = jnp.array([0.0, 1.0, 0.0, 0.0])
    img_0 = jnp.array([0.5, 0.5, 0.5, -0.5]) + 0j
    target_0 = jnp.array([1.0, 0.0, 0.0, 0.0])
    
    mesh_fn = create_engine()

    @jit
    def predict(voltages, input_vec):
        U = mesh_fn(voltages)
        return jnp.abs(jnp.dot(U, input_vec))**2

    # --- Step 1: 理想状態(無限精度)で学習 ---
    print("   1. Training Reference Model (Infinite Precision)...", end="", flush=True)
    
    @jit
    def loss_fn(params):
        p0 = predict(params, img_0); p1 = predict(params, img_1)
        return jnp.mean((p0-target_0)**2) + jnp.mean((p1-target_1)**2)

    key = jax.random.PRNGKey(42)
    params = jax.random.uniform(key, shape=(6,), minval=-0.1, maxval=0.1)
    optimizer = optax.adam(learning_rate=0.05)
    opt_state = optimizer.init(params)
    
    for i in range(800):
        grads = jax.grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
    print(" Done!")
    ideal_voltages = params
    print(f"      Ideal Voltages: {ideal_voltages}")

    # --- Step 2: 量子化実験 ---
    v_range = 2.0 
    bits_list = [2, 3, 4, 5, 6, 7, 8, 10, 12] 
    losses = []

    print("   2. Testing DAC Resolutions...")
    for bits in bits_list:
        levels = 2**bits
        step_size = v_range / levels
        quantized_voltages = jnp.round(ideal_voltages / step_size) * step_size
        l = loss_fn(quantized_voltages)
        losses.append(l)
        print(f"      {bits} bit DAC ({levels:4} levels): Loss = {l:.6f}")

    # --- グラフ化 ---
    # outputフォルダがなければ作る
    if not os.path.exists('output'):
        os.makedirs('output')

    plt.figure(figsize=(10, 6))
    plt.plot(bits_list, losses, 'o-', linewidth=3, color='purple')
    plt.title("Impact of DAC Resolution on Accuracy", fontsize=14)
    plt.xlabel("DAC Resolution (Bits)", fontsize=12)
    plt.ylabel("Classification Loss (Error)", fontsize=12)
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.axhline(y=0.01, color='red', linestyle='--', label='Acceptable Error Limit (1%)')
    plt.legend()
    
    output_path = "output/quantization_study.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Study Complete.")
    print(f"   Graph saved to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    run_quantization_study()
