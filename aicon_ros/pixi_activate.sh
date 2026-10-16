# Sourced by the pixi environments with ROS 2 (`ros`, `ros-cu126`): makes the colcon-built aicon_ros messages
# importable.

# A shell that has sourced a system-wide ROS 2, a colcon workspace or a virtual environment leaves their paths behind,
# which the environment's Python and ROS 2 must not use: keep only the entries inside the environment or the project.
_aicon_keep_own() {
    _aicon_kept=""
    _aicon_rest="$1:"
    while [ -n "${_aicon_rest}" ]; do
        _aicon_entry="${_aicon_rest%%:*}"
        _aicon_rest="${_aicon_rest#*:}"
        case "${_aicon_entry}" in
        "${CONDA_PREFIX}"* | "${PIXI_PROJECT_ROOT}"*) _aicon_kept="${_aicon_kept:+${_aicon_kept}:}${_aicon_entry}" ;;
        esac
    done
    printf '%s' "${_aicon_kept}"
}
export PYTHONPATH="$(_aicon_keep_own "${PYTHONPATH:-}")"
export AMENT_PREFIX_PATH="$(_aicon_keep_own "${AMENT_PREFIX_PATH:-}")"
export CMAKE_PREFIX_PATH="$(_aicon_keep_own "${CMAKE_PREFIX_PATH:-}")"
export COLCON_PREFIX_PATH="$(_aicon_keep_own "${COLCON_PREFIX_PATH:-}")"
export LD_LIBRARY_PATH="$(_aicon_keep_own "${LD_LIBRARY_PATH:-}")"
unset _aicon_kept _aicon_rest _aicon_entry
unset -f _aicon_keep_own

if [ -f "${PIXI_PROJECT_ROOT}/install/setup.sh" ]; then
    . "${PIXI_PROJECT_ROOT}/install/setup.sh"
fi
# The environment is activated before a task's build-ros dependency runs, so on the first build install/setup.sh does
# not exist yet: put the package's locations on the paths anyway, they are only looked up when it is imported.
AICON_ROS_PREFIX="${PIXI_PROJECT_ROOT}/install/aicon_ros"
AICON_ROS_PYTHON_VERSION="$("${CONDA_PREFIX}/bin/python" -c 'import sys; print("python%d.%d" % sys.version_info[:2])')"
AICON_ROS_PYTHON="${AICON_ROS_PREFIX}/lib/${AICON_ROS_PYTHON_VERSION}/site-packages"
export AMENT_PREFIX_PATH="${AICON_ROS_PREFIX}${AMENT_PREFIX_PATH:+:${AMENT_PREFIX_PATH}}"
export PYTHONPATH="${AICON_ROS_PYTHON}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${AICON_ROS_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# An empty XAUTHORITY makes X11 (e.g. over ssh -X) reject the connection; X then falls back to ~/.Xauthority only if
# the variable is unset, which pixi cannot do, so name that file.
if [ -n "${XAUTHORITY+set}" ] && [ -z "${XAUTHORITY}" ]; then
    export XAUTHORITY="${HOME}/.Xauthority"
fi

# Libraries from PyPI wheels (e.g. torch) find the environment's libstdc++ instead of an older system one; otherwise,
# when torch is imported before rclpy, rclpy's extension fails to load ("CXXABI_1.3.15 not found").
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
