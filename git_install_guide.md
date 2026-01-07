# Git 安装指南

为了将项目上传到 GitHub，您需要先安装 Git。以下是安装步骤：

## 1. 下载 Git

访问 [Git 官方网站](https://git-scm.com/download/win) 下载适用于 Windows 的 Git 安装程序。

## 2. 安装 Git

运行下载的安装程序，按照默认设置进行安装即可。

注意：在安装过程中，确保选择将 Git 添加到系统 PATH 环境变量中。

## 3. 验证安装

安装完成后，打开命令提示符（CMD）或 PowerShell，输入以下命令验证安装：

```bash
git --version
```

如果安装成功，会显示 Git 的版本信息。

## 4. 配置 Git

首次使用 Git 时，需要配置用户名和邮箱：

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

请将 `Your Name` 和 `your.email@example.com` 替换为您在 GitHub 上使用的用户名和邮箱。

## 5. 生成 SSH 密钥（可选但推荐）

为了更安全地与 GitHub 通信，建议生成 SSH 密钥：

```bash
ssh-keygen -t rsa -b 4096 -C "your.email@example.com"
```

然后将生成的 SSH 公钥（默认路径：`C:\Users\YourUsername\.ssh\id_rsa.pub`）添加到您的 GitHub 账户中。

## 安装完成后

安装完成后，我们可以继续将项目上传到 GitHub。
