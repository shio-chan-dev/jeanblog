# Introduction
Mermaid是一个用于使用代码创建图像的框架,今天的博客,我们将会简单介绍如何在自己的服务器上安装相关的框架,并对代码进行渲染生成图像

# 具体步骤
## 如何安装渲染框架
使用
```bash
npm install -g @mermaid-js/mermaid-cli
```
就可以安装

需要注意的是该框架使用的npm版本需要大于20,所以我们需要切换npm版本,推荐使用nvm管理npm的版本

如果没有nvm的话,使用下列命令进行安装
```bash
curl -o https://raw.githubusercontent.com/nvm-sh/nvim/v0.39.4/install.sh | bash
```
然后对shell进行重启

然后使用
```bash
nvm install 20
nvm use 20
nvm alias default 20
```
进行安装,并把默认npm切换为20

可以使用
```
node -v
npm -v
```
确认版本

## 如何进行渲染

将需要渲染的代码放置在以.mmd结尾的文件中

然后使用
```bash
mmdc -i diagrams/example.mmd -o images/example.svg
```
即可
