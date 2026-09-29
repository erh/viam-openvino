#!/bin/sh
# first_run.sh - runs once before viam-server starts the module (meta.json "first_run").
#
# Installs the host-side Intel GPU compute runtime and NPU driver that OpenVINO needs, but only
# what the detected hardware actually requires, and only on Ubuntu. Everything else is reported
# with the manual steps. It can also be run by hand at any time:  sudo ./first_run.sh
#
# Control with the module environment variable VIAM_OPENVINO_DRIVERS:
#   auto    (default) detect Intel GPU / NPU hardware and install what is missing
#   report  detect and print what is missing, install nothing
#   force   install both the GPU and NPU stacks even if no hardware was detected (containers, VMs)
#   off     do nothing
# Optional: VIAM_OPENVINO_NPU_DRIVER_VERSION=v1.19.0 pins the intel/linux-npu-driver release (default: latest).
#
# This script ALWAYS exits 0. A failing first_run aborts the machine's reconfiguration in viam-server,
# which would be far worse than running the module without an accelerator. Failures are logged and the
# module itself reports missing devices again at startup.

MODE="${VIAM_OPENVINO_DRIVERS:-auto}"
NPU_DRIVER_VERSION="${VIAM_OPENVINO_NPU_DRIVER_VERSION:-latest}"
TAG="[openvino first_run]"
log()  { echo "$TAG $*"; }
warn() { echo "$TAG WARNING: $*" >&2; }

# ------------------------------------------------------------------ platform gate
if [ "$MODE" = "off" ]; then
    log "VIAM_OPENVINO_DRIVERS=off, skipping driver setup"; exit 0
fi
OS="$(uname -s 2>/dev/null)"
ARCH="$(uname -m 2>/dev/null)"
if [ "$OS" != "Linux" ]; then
    log "not Linux ($OS); nothing to do. On Windows install the Intel graphics driver package from intel.com."; exit 0
fi
if [ "$ARCH" != "x86_64" ]; then
    log "architecture $ARCH has no Intel GPU/NPU; the CPU plugin needs no drivers. Nothing to do."; exit 0
fi

DISTRO_ID=""; DISTRO_VERSION=""; DISTRO_CODENAME=""
if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    DISTRO_ID="$ID"; DISTRO_VERSION="$VERSION_ID"; DISTRO_CODENAME="${VERSION_CODENAME:-${UBUNTU_CODENAME:-}}"
fi
log "host: $DISTRO_ID $DISTRO_VERSION ($DISTRO_CODENAME), kernel $(uname -r), mode=$MODE"

# ------------------------------------------------------------------ hardware detection
# Intel (0x8086) PCI devices: class 0x03xxxx is a display controller, class 0x1200xx is a processing
# accelerator (the NPU). Known NPU device ids are listed too in case the class field is unusual.
HAVE_GPU=0; HAVE_NPU=0; GPU_IDS=""; NPU_IDS=""
NPU_KNOWN_IDS=" 0x7d1d 0xad1d 0x643e 0xb03e 0xfd3e "   # Meteor Lake, Arrow Lake, Lunar Lake, Panther Lake, Nova Lake
for dev in /sys/bus/pci/devices/*; do
    [ -r "$dev/vendor" ] || continue
    [ "$(cat "$dev/vendor")" = "0x8086" ] || continue
    class="$(cat "$dev/class" 2>/dev/null)"; devid="$(cat "$dev/device" 2>/dev/null)"
    case "$class" in
        0x03*) HAVE_GPU=1; GPU_IDS="$GPU_IDS $devid" ;;
        0x1200*) HAVE_NPU=1; NPU_IDS="$NPU_IDS $devid" ;;
        *) case "$NPU_KNOWN_IDS" in *" $devid "*) HAVE_NPU=1; NPU_IDS="$NPU_IDS $devid" ;; esac ;;
    esac
done
if [ "$MODE" = "force" ]; then HAVE_GPU=1; HAVE_NPU=1; fi
log "Intel GPU: $([ $HAVE_GPU = 1 ] && echo "yes ($GPU_IDS )" || echo no); Intel NPU: $([ $HAVE_NPU = 1 ] && echo "yes ($NPU_IDS )" || echo no)"
if [ $HAVE_GPU = 0 ] && [ $HAVE_NPU = 0 ]; then
    log "no Intel accelerator found; the CPU plugin needs no drivers. Nothing to do."
    log "(set VIAM_OPENVINO_DRIVERS=force to install the GPU/NPU stacks anyway)"
    exit 0
fi

# ------------------------------------------------------------------ what is already installed
have_lib() { ldconfig -p 2>/dev/null | grep -q "$1" || ls /usr/lib/x86_64-linux-gnu/"$1"* /usr/lib/"$1"* >/dev/null 2>&1; }
gpu_runtime_ok() { ls /etc/OpenCL/vendors/intel*.icd >/dev/null 2>&1 || have_lib libze_intel_gpu; }
npu_runtime_ok() { have_lib libze_intel_vpu || have_lib libze_intel_npu; }
npu_kernel_ok()  { ls /dev/accel/accel* >/dev/null 2>&1; }
npu_module_available() { modinfo intel_vpu >/dev/null 2>&1; }
ze_loader_ok()   { have_lib libze_loader.so.1; }

NEED_GPU=0; NEED_NPU=0
if [ $HAVE_GPU = 1 ]; then
    if gpu_runtime_ok; then log "GPU compute runtime already installed"; else NEED_GPU=1; log "GPU compute runtime (OpenCL / Level Zero) is missing"; fi
fi
if [ $HAVE_NPU = 1 ]; then
    if npu_runtime_ok; then log "NPU user-space driver already installed"; else NEED_NPU=1; log "NPU user-space driver is missing"; fi
    if npu_kernel_ok; then
        log "NPU kernel driver is loaded (/dev/accel present)"
    elif npu_module_available; then
        warn "the intel_vpu kernel module exists but /dev/accel is absent; it may load after a reboot or the device may be disabled in firmware"
    else
        warn "no intel_vpu kernel module for kernel $(uname -r). The NPU needs kernel 6.8+ (6.11+ for Arrow Lake / Lunar Lake)."
        warn "On Ubuntu 24.04 install the HWE kernel and reboot:  sudo apt install linux-generic-hwe-24.04"
    fi
fi

if [ "$MODE" = "report" ]; then
    log "report mode: not installing anything. Run 'sudo $0' or set VIAM_OPENVINO_DRIVERS=auto to install."
    exit 0
fi
if [ $NEED_GPU = 0 ] && [ $NEED_NPU = 0 ]; then
    log "all required drivers are present"
    RUN_USER="$(id -un)"
else
    # -------------------------------------------------------------- privileges
    SUDO=""
    if [ "$(id -u)" != "0" ]; then
        if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
            SUDO="sudo"
        else
            warn "not running as root and passwordless sudo is unavailable; cannot install drivers automatically."
            warn "Run:  sudo $(cd "$(dirname "$0")" && pwd)/first_run.sh"
            exit 0
        fi
    fi
    RUN_USER="$(id -un)"

    if [ "$DISTRO_ID" != "ubuntu" ]; then
        warn "automatic installation is only implemented for Ubuntu (found '$DISTRO_ID'). See the README for manual steps:"
        warn "  GPU: Intel compute runtime (intel-opencl-icd, Level Zero GPU)   NPU: https://github.com/intel/linux-npu-driver/releases"
        exit 0
    fi
    export DEBIAN_FRONTEND=noninteractive
    APT="$SUDO apt-get -qq -o Dpkg::Use-Pty=0"
    apt_updated=0
    apt_update() { if [ $apt_updated = 0 ]; then $APT update >/dev/null 2>&1 || warn "apt-get update reported errors"; apt_updated=1; fi; }
    apt_install() { apt_update; $APT install -y "$@" >/dev/null 2>&1; }
    fetch() { # fetch URL OUT
        if command -v curl >/dev/null 2>&1; then curl -fsSL --retry 3 -o "$2" "$1"
        elif command -v wget >/dev/null 2>&1; then wget -q -O "$2" "$1"
        else return 1; fi
    }
    if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
        apt_install curl ca-certificates || warn "could not install curl"
    fi

    # -------------------------------------------------------------- Intel graphics apt repository
    intel_repo_added=0
    add_intel_repo() {
        case "$DISTRO_CODENAME" in
            jammy|noble) ;;
            *) return 1 ;;   # no Intel client repo for this release; fall back to distro packages
        esac
        [ $intel_repo_added = 1 ] && return 0
        apt_install gpg ca-certificates >/dev/null 2>&1
        if fetch https://repositories.intel.com/gpu/intel-graphics.key /tmp/intel-graphics.key \
           && $SUDO gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg /tmp/intel-graphics.key 2>/dev/null; then
            echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu $DISTRO_CODENAME client" \
                | $SUDO tee /etc/apt/sources.list.d/intel-gpu-"$DISTRO_CODENAME".list >/dev/null
            apt_updated=0; apt_update; intel_repo_added=1; return 0
        fi
        return 1
    }

    # -------------------------------------------------------------- GPU
    if [ $NEED_GPU = 1 ]; then
        log "installing Intel GPU compute runtime..."
        ok=0
        if add_intel_repo; then
            case "$DISTRO_CODENAME" in
                noble) apt_install libze-intel-gpu1 libze1 intel-opencl-icd clinfo && ok=1 ;;
                jammy) apt_install intel-opencl-icd intel-level-zero-gpu level-zero clinfo && ok=1 ;;
            esac
        fi
        if [ $ok = 0 ]; then
            log "Intel repository unavailable for '$DISTRO_CODENAME'; trying Ubuntu's own packages"
            apt_install intel-opencl-icd clinfo && ok=1
            apt_install libze-intel-gpu1 libze1 >/dev/null 2>&1 || true
        fi
        if [ $ok = 1 ] && gpu_runtime_ok; then log "GPU compute runtime installed"; else warn "GPU compute runtime installation failed; see README for manual steps"; fi
    fi

    # -------------------------------------------------------------- NPU
    if [ $NEED_NPU = 1 ]; then
        log "installing Intel NPU user-space driver ($NPU_DRIVER_VERSION) from github.com/intel/linux-npu-driver..."
        if [ "$NPU_DRIVER_VERSION" = "latest" ]; then api="https://api.github.com/repos/intel/linux-npu-driver/releases/latest"
        else api="https://api.github.com/repos/intel/linux-npu-driver/releases/tags/$NPU_DRIVER_VERSION"; fi
        tmpd="$(mktemp -d /tmp/npu-driver.XXXXXX)"
        if fetch "$api" "$tmpd/release.json"; then
            urls_of() { grep -o '"browser_download_url": *"[^"]*'"$1"'"' "$tmpd/release.json" | sed 's/.*"\(http[^"]*\)"/\1/'; }
            # Current releases ship one tarball of .deb packages per Ubuntu release
            # (linux-npu-driver-<ver>-ubuntu2404.tar.gz); older releases attached the .deb files directly.
            short="$(echo "$DISTRO_VERSION" | tr -d .)"
            urls="$(urls_of "ubuntu${short}.tar.gz")"
            [ -z "$urls" ] && urls="$(urls_of "ubuntu${DISTRO_VERSION}_amd64.deb")"
            if [ -z "$urls" ]; then
                avail="$(grep -o '"name": *"[^"]*ubuntu[0-9.]*[^"]*"' "$tmpd/release.json" | grep -o 'ubuntu[0-9.]*' | sort -u | tr '\n' ' ')"
                warn "this NPU driver release has no build for Ubuntu $DISTRO_VERSION (available: ${avail:-none})."
                warn "Pin an older release with VIAM_OPENVINO_NPU_DRIVER_VERSION=vX.Y.Z or install manually from $api"
            else
                got=0
                for u in $urls; do
                    f="$tmpd/$(basename "$u")"
                    if fetch "$u" "$f"; then got=$((got+1)); else warn "download failed: $u"; fi
                done
                for t in "$tmpd"/*.tar.gz; do [ -f "$t" ] && tar -xzf "$t" -C "$tmpd" 2>/dev/null; done
                # Skip debug-symbol packages; install the driver, compiler and firmware packages.
                debs="$(find "$tmpd" -name '*.deb' ! -name '*dbgsym*' | tr '\n' ' ')"
                if [ $got -gt 0 ] && [ -n "$debs" ]; then
                    # Level Zero loader first (libze1 from the Intel/Ubuntu repos, or a loader .deb shipped with the release).
                    if ! ze_loader_ok; then
                        add_intel_repo >/dev/null 2>&1
                        apt_install libze1 >/dev/null 2>&1 || apt_install level-zero >/dev/null 2>&1 || true
                    fi
                    apt_install libtbb12 >/dev/null 2>&1 || true
                    # shellcheck disable=SC2086
                    if $SUDO dpkg -i $debs >/dev/null 2>&1 || $APT install -f -y >/dev/null 2>&1; then
                        log "NPU driver packages installed: $(for d in $debs; do basename "$d" | cut -d_ -f1; done | tr '\n' ' ')"
                    else
                        warn "dpkg reported errors installing the NPU driver packages"
                    fi
                    # Device node access for non-root viam-server users.
                    if [ ! -f /etc/udev/rules.d/10-intel-vpu.rules ]; then
                        $SUDO mkdir -p /etc/udev/rules.d
                        echo 'SUBSYSTEM=="accel", KERNEL=="accel*", GROUP="render", MODE="0660"' | $SUDO tee /etc/udev/rules.d/10-intel-vpu.rules >/dev/null
                        $SUDO udevadm control --reload-rules >/dev/null 2>&1; $SUDO udevadm trigger --subsystem-match=accel >/dev/null 2>&1
                    fi
                else
                    warn "no NPU driver .deb packages could be downloaded"
                fi
            fi
        else
            warn "could not reach the GitHub releases API; NPU driver not installed. Manual steps: https://github.com/intel/linux-npu-driver/releases"
        fi
        rm -rf "$tmpd"
        if npu_runtime_ok; then log "NPU user-space driver installed"; else warn "NPU user-space driver is still missing"; fi
    fi
fi

# ------------------------------------------------------------------ device access for a non-root viam-server
if [ "$RUN_USER" != "root" ]; then
    for grp in render video; do
        if getent group "$grp" >/dev/null 2>&1 && ! id -nG "$RUN_USER" | tr ' ' '\n' | grep -qx "$grp"; then
            if ${SUDO:-} usermod -aG "$grp" "$RUN_USER" 2>/dev/null; then
                log "added user $RUN_USER to group $grp (takes effect after viam-server restarts)"
            else
                warn "could not add $RUN_USER to group $grp; run: sudo usermod -aG $grp $RUN_USER"
            fi
        fi
    done
fi

# ------------------------------------------------------------------ summary
log "summary: GPU runtime $(gpu_runtime_ok && echo present || echo missing); NPU driver $(npu_runtime_ok && echo present || echo missing); /dev/accel $(npu_kernel_ok && echo present || echo absent)"
log "restart viam-server (or reboot if a kernel module was just loaded), then confirm with DoCommand list_devices on erh:openvino:diagnostics"
exit 0
