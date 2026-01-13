#!/bin/bash
#===============================================================================
# BindCraft MCP Quick Setup Script
#===============================================================================
# This script sets up the complete environment for BindCraft MCP server.
#
# After cloning the repository, run this script to set everything up:
#   git clone <repository_url> bindcraft_mcp
#   cd bindcraft_mcp
#   bash quick_setup.sh
#
# Once setup is complete, register in Claude Code with the config shown at the end.
#
# Options:
#   --cuda VERSION    CUDA version (e.g., '12.4'). Auto-detect if not specified.
#   --pkg-manager     Package manager: 'mamba' or 'conda' (default: auto-detect)
#   --env-path PATH   Environment path (default: ./env)
#   --skip-weights    Skip downloading AlphaFold2 weights (~5.3 GB)
#   --skip-repo       Skip cloning BindCraft repository
#   --help            Show this help message
#
# Example:
#   bash quick_setup.sh --cuda 12.4 --pkg-manager mamba
#===============================================================================

# Prevent running with Python
if [ -n "$PYTHON_VERSION" ] || [ "$(basename "$0")" = "python" ] || [ "$(basename "$0")" = "python3" ]; then
    echo "ERROR: This is a bash script. Run it with: bash quick_setup.sh"
    exit 1
fi

# Check if being sourced or executed
if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
    echo "ERROR: This script should be executed, not sourced."
    echo "Run it with: bash quick_setup.sh"
    return 1
fi

set -e  # Exit on error

#===============================================================================
# Colors for output
#===============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

#===============================================================================
# Default values
#===============================================================================
CUDA_VERSION=""
PKG_MANAGER=""
ENV_PATH="./env"
SKIP_WEIGHTS=false
SKIP_REPO=false

# Installation directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${SCRIPT_DIR}/repo"
BINDCRAFT_REPO_DIR="${REPO_DIR}/BindCraft"
SCRIPTS_DIR="${REPO_DIR}/scripts"

#===============================================================================
# Helper Functions
#===============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
}

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

show_help() {
    echo "BindCraft MCP Quick Setup Script"
    echo ""
    echo "Usage: bash quick_setup.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --cuda VERSION      CUDA version (e.g., '12.4'). Auto-detect if not specified."
    echo "  --pkg-manager NAME  Package manager: 'mamba' or 'conda' (default: auto-detect)"
    echo "  --env-path PATH     Environment path (default: ./env)"
    echo "  --skip-weights      Skip downloading AlphaFold2 weights (~5.3 GB)"
    echo "  --skip-repo         Skip cloning BindCraft repository"
    echo "  --help, -h          Show this help message"
    echo ""
    echo "Example:"
    echo "  bash quick_setup.sh --cuda 12.4 --pkg-manager mamba"
    exit 0
}

#===============================================================================
# Parse Command Line Arguments
#===============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --pkg-manager|--pkg_manager)
            PKG_MANAGER="$2"
            shift 2
            ;;
        --env-path|--env_path)
            ENV_PATH="$2"
            shift 2
            ;;
        --skip-weights)
            SKIP_WEIGHTS=true
            shift
            ;;
        --skip-repo)
            SKIP_REPO=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

#===============================================================================
# Start Installation
#===============================================================================

print_header "BindCraft MCP Quick Setup"
SECONDS=0

print_info "Installation directory: ${SCRIPT_DIR}"
print_info "Environment path: ${ENV_PATH}"

#===============================================================================
# Step 1: Check Prerequisites
#===============================================================================

print_step "Checking prerequisites..."

# Check for conda installation
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed or not in PATH."
    print_info "Please install Miniconda or Anaconda first:"
    print_info "  https://docs.conda.io/en/latest/miniconda.html"
    print_info "  or"
    print_info "  https://github.com/conda-forge/miniforge (recommended)"
    exit 1
fi

CONDA_BASE=$(conda info --base 2>/dev/null) || {
    print_error "Could not determine conda base directory"
    exit 1
}
print_info "Conda found at: ${CONDA_BASE}"

# Auto-detect package manager if not specified
if [ -z "$PKG_MANAGER" ]; then
    if command -v mamba &> /dev/null; then
        PKG_MANAGER="mamba"
        print_info "Using mamba (faster than conda)"
    else
        PKG_MANAGER="conda"
        print_info "Using conda (install mamba for faster setup)"
    fi
else
    print_info "Using package manager: ${PKG_MANAGER}"
fi

# Verify package manager exists
if ! command -v $PKG_MANAGER &> /dev/null; then
    print_error "${PKG_MANAGER} not found. Please install it or use --pkg-manager to specify another."
    exit 1
fi

# Check for git
if ! command -v git &> /dev/null; then
    print_error "git is not installed. Please install git first."
    exit 1
fi

# Check for wget or curl
if command -v wget &> /dev/null; then
    DOWNLOAD_CMD="wget -O"
elif command -v curl &> /dev/null; then
    DOWNLOAD_CMD="curl -L -o"
else
    print_error "Neither wget nor curl is installed. Please install one of them."
    exit 1
fi

# Auto-detect CUDA version if not specified
if [ -z "$CUDA_VERSION" ]; then
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
        print_info "Auto-detected CUDA version: ${CUDA_VERSION}"
    elif command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed -n 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/p')
        print_info "Auto-detected CUDA version: ${CUDA_VERSION}"
    else
        print_warning "Could not detect CUDA version. Installing CPU-only version."
        print_warning "For GPU support, re-run with: --cuda VERSION"
    fi
else
    print_info "Using specified CUDA version: ${CUDA_VERSION}"
fi

print_success "Prerequisites check passed"

#===============================================================================
# Step 2: Clone BindCraft Repository
#===============================================================================

if [ "$SKIP_REPO" = false ]; then
    print_step "Setting up BindCraft repository..."

    mkdir -p "${REPO_DIR}"

    if [ -d "${BINDCRAFT_REPO_DIR}" ] && [ -f "${BINDCRAFT_REPO_DIR}/bindcraft.py" ]; then
        print_info "BindCraft repository already exists"
        cd "${BINDCRAFT_REPO_DIR}"
        git pull 2>/dev/null || print_warning "Could not pull updates (offline?)"
        cd "${SCRIPT_DIR}"
    else
        print_info "Cloning BindCraft repository..."
        rm -rf "${BINDCRAFT_REPO_DIR}" 2>/dev/null || true
        git clone --depth 1 https://github.com/martinpacesa/BindCraft "${BINDCRAFT_REPO_DIR}" || {
            print_error "Failed to clone BindCraft repository"
            exit 1
        }
    fi

    # Setup scripts directory
    mkdir -p "${SCRIPTS_DIR}"
    mkdir -p "${SCRIPTS_DIR}/functions"

    # Copy essential files
    if [ ! -f "${SCRIPTS_DIR}/run_bindcraft.py" ]; then
        cp "${BINDCRAFT_REPO_DIR}/bindcraft.py" "${SCRIPTS_DIR}/run_bindcraft.py"
        print_info "Copied bindcraft.py to scripts/"
    fi

    if [ ! -f "${SCRIPTS_DIR}/functions/dssp" ]; then
        cp -r "${BINDCRAFT_REPO_DIR}/functions/"* "${SCRIPTS_DIR}/functions/"
        print_info "Copied functions to scripts/"
    fi

    print_success "BindCraft repository setup complete"
else
    print_info "Skipping repository clone (--skip-repo)"
fi

#===============================================================================
# Step 3: Create Conda Environment
#===============================================================================

print_step "Setting up conda environment..."

# Convert to absolute path
if [[ "${ENV_PATH}" != /* ]]; then
    ENV_PATH="${SCRIPT_DIR}/${ENV_PATH#./}"
fi

SKIP_ENV_CREATE=false

if [ -d "${ENV_PATH}" ] && [ -f "${ENV_PATH}/bin/python" ]; then
    print_info "Environment exists at ${ENV_PATH}"

    # Quick check for key packages
    if "${ENV_PATH}/bin/python" -c "import fastmcp; import jax" 2>/dev/null; then
        print_info "Environment appears functional. Skipping recreation."
        SKIP_ENV_CREATE=true
    else
        print_warning "Environment incomplete. Recreating..."
    fi
fi

if [ "$SKIP_ENV_CREATE" = false ]; then
    # Remove existing broken environment
    if [ -d "${ENV_PATH}" ]; then
        print_info "Removing incomplete environment..."
        rm -rf "${ENV_PATH}"
    fi

    print_info "Creating new environment with Python 3.10..."
    print_info "This may take 10-30 minutes..."

    $PKG_MANAGER create -p "${ENV_PATH}" python=3.10 -y || {
        print_error "Failed to create conda environment"
        exit 1
    }

    # Install packages in batches for better error handling
    print_info "Installing core packages..."
    $PKG_MANAGER install -p "${ENV_PATH}" \
        pip pandas matplotlib 'numpy<2.0.0' biopython scipy seaborn \
        libgfortran5 tqdm ffmpeg fsspec \
        -c conda-forge -y || {
        print_error "Failed to install core packages"
        exit 1
    }

    print_info "Installing ML packages..."
    $PKG_MANAGER install -p "${ENV_PATH}" \
        chex dm-haiku 'flax<0.10.0' dm-tree joblib ml-collections immutabledict optax \
        -c conda-forge -y || {
        print_error "Failed to install ML packages"
        exit 1
    }

    print_info "Installing JAX..."
    if [ -n "$CUDA_VERSION" ]; then
        CONDA_OVERRIDE_CUDA="$CUDA_VERSION" $PKG_MANAGER install -p "${ENV_PATH}" \
            'jax>=0.4,<=0.6.0' 'jaxlib>=0.4,<=0.6.0=*cuda*' \
            -c conda-forge -c nvidia -y || {
            print_warning "CUDA JAX failed, trying CPU version..."
            $PKG_MANAGER install -p "${ENV_PATH}" \
                'jax>=0.4,<=0.6.0' 'jaxlib>=0.4,<=0.6.0' \
                -c conda-forge -y
        }
    else
        $PKG_MANAGER install -p "${ENV_PATH}" \
            'jax>=0.4,<=0.6.0' 'jaxlib>=0.4,<=0.6.0' \
            -c conda-forge -y || {
            print_error "Failed to install JAX"
            exit 1
        }
    fi

    print_info "Installing PyRosetta (may take a while)..."
    $PKG_MANAGER install -p "${ENV_PATH}" pyrosetta pdbfixer \
        --channel https://conda.graylab.jhu.edu -c conda-forge -y 2>/dev/null || {
        print_warning "PyRosetta installation failed (license may be required)"
        print_warning "Some features may not work without PyRosetta"
    }

    print_info "Installing MCP packages..."
    "${ENV_PATH}/bin/pip" install --quiet fastmcp loguru click || {
        print_error "Failed to install MCP packages"
        exit 1
    }

    print_info "Installing ColabDesign..."
    "${ENV_PATH}/bin/pip" install --quiet git+https://github.com/sokrypton/ColabDesign.git --no-deps || {
        print_warning "ColabDesign installation failed"
    }
fi

print_success "Conda environment ready at ${ENV_PATH}"

#===============================================================================
# Step 4: Download AlphaFold2 Weights
#===============================================================================

if [ "$SKIP_WEIGHTS" = false ]; then
    print_step "Setting up AlphaFold2 model weights..."

    PARAMS_DIR="${SCRIPTS_DIR}/params"

    if [ -f "${PARAMS_DIR}/params_model_5_ptm.npz" ]; then
        print_info "AlphaFold2 weights already present"
    else
        print_info "Downloading AlphaFold2 weights (~5.3 GB)..."
        print_info "This may take 10-30 minutes depending on your connection..."

        mkdir -p "${PARAMS_DIR}"

        PARAMS_TAR="${PARAMS_DIR}/alphafold_params.tar"
        $DOWNLOAD_CMD "${PARAMS_TAR}" "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar" || {
            print_error "Failed to download AlphaFold2 weights"
            exit 1
        }

        print_info "Extracting weights..."
        tar -xf "${PARAMS_TAR}" -C "${PARAMS_DIR}" || {
            print_error "Failed to extract weights"
            exit 1
        }

        rm -f "${PARAMS_TAR}"
        print_success "AlphaFold2 weights ready"
    fi
else
    print_info "Skipping AlphaFold2 weights (--skip-weights)"
fi

#===============================================================================
# Step 5: Set Permissions
#===============================================================================

print_step "Setting permissions..."

for exe in "${SCRIPTS_DIR}/functions/dssp" "${SCRIPTS_DIR}/functions/DAlphaBall.gcc" \
           "${BINDCRAFT_REPO_DIR}/functions/dssp" "${BINDCRAFT_REPO_DIR}/functions/DAlphaBall.gcc"; do
    if [ -f "$exe" ]; then
        chmod +x "$exe" 2>/dev/null || true
    fi
done

print_success "Permissions set"

#===============================================================================
# Step 6: Verify Installation
#===============================================================================

print_step "Verifying installation..."

# Test core imports
"${ENV_PATH}/bin/python" << 'PYTEST'
import sys
failed = []

for pkg in ['fastmcp', 'loguru', 'jax', 'numpy', 'pandas']:
    try:
        __import__(pkg)
        print(f'  [OK] {pkg}')
    except ImportError as e:
        failed.append(pkg)
        print(f'  [FAIL] {pkg}: {e}')

try:
    from Bio import PDB
    print('  [OK] biopython')
except ImportError:
    failed.append('biopython')
    print('  [FAIL] biopython')

if failed:
    print(f'\nWarning: {len(failed)} packages failed to import')
    sys.exit(1)
else:
    print('\nCore packages OK!')
PYTEST

# Test MCP server
print_info "Testing MCP server..."
"${ENV_PATH}/bin/python" -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}/src')
try:
    from bindcraft_mcp import mcp
    print('  [OK] BindCraft MCP server loads successfully')
except Exception as e:
    print(f'  [FAIL] MCP server: {e}')
    sys.exit(1)
" || print_warning "MCP server test failed"

print_success "Verification complete"

#===============================================================================
# Step 7: Cleanup
#===============================================================================

print_step "Cleaning up..."
$PKG_MANAGER clean -a -y 2>/dev/null || true
print_success "Cleanup complete"

#===============================================================================
# Generate Claude Code Config
#===============================================================================

print_header "Setup Complete!"

echo -e "${GREEN}BindCraft MCP is ready to use!${NC}"
echo ""
echo "Environment: ${ENV_PATH}"
echo "MCP Server:  ${SCRIPT_DIR}/src/bindcraft_mcp.py"
echo ""

# Generate Claude Code config
CONFIG_JSON=$(cat << EOF
{
  "mcpServers": {
    "bindcraft": {
      "command": "${ENV_PATH}/bin/python",
      "args": ["${SCRIPT_DIR}/src/bindcraft_mcp.py"],
      "env": {
        "PYTHONPATH": "${SCRIPT_DIR}/src:${SCRIPT_DIR}/clean_scripts"
      }
    }
  }
}
EOF
)

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}  Claude Code Configuration${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Add this to your Claude Code MCP settings:"
echo ""
echo -e "${YELLOW}${CONFIG_JSON}${NC}"
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Also save config to file
CONFIG_FILE="${SCRIPT_DIR}/claude_code_config.json"
echo "${CONFIG_JSON}" > "${CONFIG_FILE}"
echo -e "Config saved to: ${YELLOW}${CONFIG_FILE}${NC}"
echo ""

# Manual run instructions
echo "To test the MCP server manually:"
echo -e "  ${YELLOW}${ENV_PATH}/bin/python ${SCRIPT_DIR}/src/bindcraft_mcp.py${NC}"
echo ""

t=$SECONDS
echo "Setup completed in $(($t / 60)) minutes and $(($t % 60)) seconds."
