#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=""
INSTALL_GLOBAL=1
WITH_TEST=0

while (($#)); do
    case "$1" in
        --python)
            [[ $# -ge 2 ]] || { echo "--python 需要路径" >&2; exit 2; }
            PYTHON_BIN=$2
            shift 2
            ;;
        --no-global)
            INSTALL_GLOBAL=0
            shift
            ;;
        --with-test)
            WITH_TEST=1
            shift
            ;;
        -h|--help)
            echo "用法: ./install.sh [--python /path/to/python] [--no-global] [--with-test]"
            exit 0
            ;;
        *)
            echo "未知参数: $1" >&2
            exit 2
            ;;
    esac
done

has_deepmd() {
    "$1" -c 'import deepmd, ase, yaml' >/dev/null 2>&1
}

if [[ -z "$PYTHON_BIN" ]]; then
    ACTIVE_PYTHON=$(command -v python || true)
    if [[ -n "$ACTIVE_PYTHON" ]] && has_deepmd "$ACTIVE_PYTHON"; then
        PYTHON_BIN=$ACTIVE_PYTHON
    elif [[ -x /opt/deepmd-kit/.venv/bin/python ]] && has_deepmd /opt/deepmd-kit/.venv/bin/python; then
        PYTHON_BIN=/opt/deepmd-kit/.venv/bin/python
    else
        echo "未找到可导入 DeepMD 的 Python。请先准备 DeepMD/Torch 环境，再使用 --python 指定。" >&2
        exit 2
    fi
fi

[[ -x "$PYTHON_BIN" ]] || { echo "Python 不可执行: $PYTHON_BIN" >&2; exit 2; }
has_deepmd "$PYTHON_BIN" || { echo "指定 Python 无法导入 deepmd/ase/yaml: $PYTHON_BIN" >&2; exit 2; }

(cd "$PROJECT_DIR/models" && sha256sum -c dpa4_omat_neo704.pt2.sha256)

if ((WITH_TEST)); then
    "$PYTHON_BIN" -m pip install -e "$PROJECT_DIR[test]"
else
    "$PYTHON_BIN" -m pip install -e "$PROJECT_DIR"
fi

if ((INSTALL_GLOBAL)); then
    LAUNCHER_DIR=/usr/local/bin
    [[ -w "$LAUNCHER_DIR" ]] || { echo "$LAUNCHER_DIR 不可写；请用 root/sudo 安装，或添加 --no-global。" >&2; exit 2; }
    LAUNCHER_TMP=$(mktemp)
    trap 'rm -f "$LAUNCHER_TMP"' EXIT
    printf '#!/usr/bin/env bash\nexec %q -m phonon_kit.cli "$@"\n' "$PYTHON_BIN" > "$LAUNCHER_TMP"
    install -m 0755 "$LAUNCHER_TMP" "$LAUNCHER_DIR/ph"
fi

echo "安装完成"
echo "Python: $PYTHON_BIN"
if ((INSTALL_GLOBAL)); then
    echo "全局命令: /usr/local/bin/ph"
else
    echo "命令: $PYTHON_BIN -m phonon_kit.cli"
fi
echo "快速开始: ph init my_phonon_case"

