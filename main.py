"""共享单车数据下载主程序。

直接运行：
    python3 main.py

自定义数量：
    python3 main.py --limit 100000
"""

from __future__ import annotations

import os

from config_local import APP_KEY
from download_shenzhen_bikes import main as download_main


def main() -> None:
    # 让下载程序从本地配置读取密钥，用户无需在终端输入 appKey。
    os.environ["SZ_APP_KEY"] = APP_KEY
    download_main()


if __name__ == "__main__":
    main()
