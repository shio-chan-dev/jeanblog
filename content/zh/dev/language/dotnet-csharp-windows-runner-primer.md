---
title: "从 Python/Linux 到 C#/.NET：为 Windows Runner 建立第一套心智模型"
subtitle: "理解语言、运行平台和项目工具链，再创建第一个可运行的 .NET 8 控制台程序"
date: 2026-08-26T16:00:00+08:00
draft: false
summary: "面向熟悉 Python/Linux、但第一次接触 C# 的开发者，从 Windows Runner 场景出发解释 .NET、C#、CLR、SDK、NuGet 和 csproj，并在 WSL 中完成一个最小控制台项目。"
description: "从 Windows Runner 场景入门 C# 与 .NET 8：理解 SDK、Runtime、CLR、NuGet 和 csproj，在 Ubuntu/WSL 中创建、编译并运行第一个控制台项目，并分清 Linux 开发与 Windows UKey 验收的边界。"
categories: ["语言设计"]
tags: ["C#", ".NET 8", "CLR", "NuGet", "WSL", "Windows Runner"]
keywords: ["C# 入门", ".NET 入门", ".NET 8", "dotnet CLI", "NuGet", "csproj", "Windows Runner", "WSL"]
readingTime: 16
---

如果你主要使用 Python 和 Linux，第一次看到下面这组命令，很容易把 `.NET`、C#、`dotnet` 和 `csproj` 当成同一个东西：

```bash
dotnet new console --framework net8.0 --output WindowsRunner
dotnet add WindowsRunner/WindowsRunner.csproj package Microsoft.Playwright
dotnet build WindowsRunner/WindowsRunner.csproj
```

它们其实分属不同层次：C# 是编程语言，.NET 是编译和运行程序的平台，`dotnet` 是操作这个平台的命令行入口，`WindowsRunner.csproj` 则描述一个具体项目。

本文从一个真实需求出发：我们准备编写一个安装在用户 Windows 电脑上的轻量 Runner，未来由它启动系统 Chrome，并让用户在本机完成 UKey 登录。但在碰浏览器、证书和 Windows API 之前，先把问题压缩成一个更小、仍然完整的学习目标：

> 在 Ubuntu/WSL 中安装 .NET 8 SDK，创建一个 C# 控制台项目，让它接收一段命令行输入，并能够被编译和运行。

完成这一步后，你应该能回答四个问题：

1. `.NET` 和 C# 分别是什么？
2. SDK、Runtime、CLR、NuGet 和 `csproj` 各自负责什么？
3. `dotnet new`、`dotnet build` 和 `dotnet run` 实际做了什么？
4. 为什么代码可以在 Linux 编写，但 UKey PoC 必须在原生 Windows 进程中验收？

本文不会实现 Playwright 浏览器控制、UKey、WebSocket、FastAPI、数据库或安装包。这些能力依赖当前基础，但不应该在第一次接触 .NET 时同时引入。

## 先建立总图：C# 是语言，.NET 是平台

最短的定义是：

- **C#** 是描述程序逻辑的编程语言。
- **.NET** 是用于构建和运行应用的平台。
- **`dotnet`** 是 .NET SDK 提供的命令行工具。

它们之间的关系可以画成一条流水线：

```text
Program.cs 中的 C# 源代码
            |
            v
       .NET SDK 编译
            |
            v
  IL（中间语言）+ 类型元数据
            |
            v
      .NET Runtime / CLR
            |
            v
 Windows、Linux 或 macOS 上的机器指令
```

C# 编译器通常不会直接把每一行代码固定成某一种 CPU 的机器码，而是先生成包含中间语言和类型元数据的程序集。运行时，CLR（Common Language Runtime，公共语言运行时）负责加载程序集、管理内存和异常，并通过 JIT（Just-In-Time）编译等机制执行代码。

.NET 也支持 ReadyToRun 和 Native AOT 等发布方式，但它们不是入门阶段的重点。当前只需要记住：**C# 负责表达逻辑，.NET 负责把逻辑构建并运行起来。**

### 如果你来自 Python

下面的类比有助于快速定位概念，但它不是严格的一一对应：

| .NET 世界 | 主要职责 | Python 世界中的近似概念 |
| --- | --- | --- |
| C# | 编程语言 | Python 语言 |
| .NET SDK | 创建、还原依赖、编译、测试和发布项目 | Python + uv/pip + 构建工具 |
| .NET Runtime | 运行已经构建的 .NET 应用 | Python 解释器 |
| CLR | 加载代码、执行 IL、垃圾回收、异常和类型安全 | CPython 运行时承担的部分职责 |
| NuGet | 查找和分发依赖包 | PyPI |
| `WindowsRunner.csproj` | 项目目标、构建设置和依赖声明 | `pyproject.toml` |
| `Program.cs` | 默认程序入口 | `main.py` |

最容易犯的错误，是把 `.NET` 说成一种语言。更准确的说法是：

> 我们使用 C# 编写代码，使用 .NET 8 构建和运行程序。

## SDK、Runtime、CLR 和基础类库分别做什么

`.NET` 这个名字覆盖了一组协作组件。先理解其中四个就够了。

### SDK：开发时使用的完整工具箱

SDK 是 Software Development Kit 的缩写。安装 `.NET 8 SDK` 后，你会得到：

- C# 编译器；
- `dotnet` 命令行工具；
- 项目模板；
- NuGet 依赖还原能力；
- 构建、测试和发布工具；
- 对应版本的 .NET Runtime。

因此，开发电脑通常安装 SDK。只运行现成应用的电脑可以只安装 Runtime；以后也可以把 Runtime 一起打包，发布成 self-contained 应用。

### Runtime：运行已构建应用所需的环境

Runtime 不负责创建项目。它负责加载并执行已经构建好的 .NET 应用。

一个常见判断是：

```text
需要写代码、编译和测试 -> 安装 SDK
只需要运行已有程序     -> Runtime 可能已经足够
```

### CLR：Runtime 中的执行核心

CLR 主要负责：

- 加载程序集；
- 执行中间语言；
- 垃圾回收；
- 类型安全；
- 异常处理；
- 线程和运行时服务。

对于 Python 开发者，最值得注意的是类型检查时机不同。Python 的许多类型错误要等相应代码路径真正运行时才暴露；C# 的大量类型错误会在 `dotnet build` 阶段被编译器发现。

### 基础类库：不必从零实现常用能力

.NET 自带大量标准 API，例如：

- `Console`：控制台输入输出；
- `String`：字符串；
- `List<T>` 和 `Dictionary<TKey, TValue>`：集合；
- `File` 和 `Directory`：文件系统；
- `HttpClient`：HTTP 请求；
- `Task`：异步操作；
- `Uri`：网址解析。

这些类型属于 .NET 基础类库。C# 是调用它们的语言，而不是它们本身。

## 第一个检查点：在 WSL 中得到可用的 SDK

当前场景是 Ubuntu 24.04 on WSL2。Ubuntu 软件源已经提供 `.NET 8 SDK`，因此先使用发行版包，不额外混入 Snap 或另一个软件源。

```bash
sudo apt update
sudo apt install -y dotnet-sdk-8.0
```

安装完成后先不要急着创建项目，先确认真正被调用的是哪个 SDK：

```bash
dotnet --version
dotnet --list-sdks
dotnet --info
```

三个命令回答的问题不同：

| 命令 | 回答的问题 |
| --- | --- |
| `dotnet --version` | 当前命令默认选择了哪个 SDK？ |
| `dotnet --list-sdks` | 这台机器安装了哪些 SDK？ |
| `dotnet --info` | SDK、Runtime、操作系统和 CPU 架构分别是什么？ |

### Checkpoint

继续之前，至少确认：

```text
SDK 主版本为 8
操作系统被识别为 Linux
Architecture 为 x64（当前机器）
```

如果终端提示 `dotnet: command not found`，说明当前 shell 还找不到 SDK。此时应该先检查安装结果和 `PATH`，不要通过手写项目文件绕过环境问题。

## 第二个检查点：从外部行为创建第一个项目

我们的最小外部行为是：运行程序，看到一行确定的输出。暂时不需要类、配置对象、状态机或浏览器适配器，因为当前行为还没有产生这些结构的必要性。

在准备存放代码的目录执行：

```bash
dotnet new console --framework net8.0 --output windows-runner/src/WindowsRunner
```

这条命令可以从右向左理解：

- `--output windows-runner/src/WindowsRunner`：把项目生成到指定目录；
- `--framework net8.0`：项目面向 .NET 8；
- `console`：采用控制台程序模板；
- `dotnet new`：从模板创建新项目。

生成后的核心结构是：

```text
windows-runner/
└── src/
    └── WindowsRunner/
        ├── Program.cs
        ├── WindowsRunner.csproj
        └── obj/
```

`Program.cs` 默认只有一行核心代码：

```csharp
Console.WriteLine("Hello, World!");
```

运行它：

```bash
dotnet run --project windows-runner/src/WindowsRunner/WindowsRunner.csproj
```

预期输出：

```text
Hello, World!
```

这个版本很小，但它已经走通了一条完整工程链：模板生成源文件，SDK 还原项目，编译器编译 C#，Runtime 加载并执行程序集。

### 为什么没有看到 `Main` 方法

现代 C# 控制台模板默认使用 **top-level statements（顶级语句）**。编译器会为这段顶级代码生成程序入口，所以不必一开始就写出：

```csharp
internal class Program
{
    private static void Main(string[] args)
    {
        Console.WriteLine("Hello, World!");
    }
}
```

两种写法都能表达入口。顶级语句只是减少第一个小程序的固定样板，不代表 C# 没有类或 `Main` 方法模型。

### Checkpoint

到这里，项目已经能稳定输出 `Hello, World!`。它仍然没有接收任何外部输入，所以下一步让调用者决定这次准备执行什么任务。

## 第三个检查点：让程序接收一个任务名称

如果程序永远输出固定字符串，那么无论调用者输入什么，它都无法区分不同任务。这个限制可以通过一个最小压力示例看出来：

```bash
dotnet run --project windows-runner/src/WindowsRunner/WindowsRunner.csproj -- "Chrome 启动检查"
```

当前程序仍然只会输出 `Hello, World!`。`--` 后面的内容已经由 `dotnet run` 传给应用，但 `Program.cs` 没有读取它。

把 `Program.cs` 替换为：

```csharp
var taskName = args.Length > 0 ? args[0] : "Windows Runner PoC";

Console.WriteLine($"准备任务：{taskName}");
```

这里一次引入了几条常用 C# 规则：

- 语句通常以分号 `;` 结束；
- `var` 让编译器从右侧表达式推断静态类型；
- `args` 是传给程序的字符串数组；
- `condition ? a : b` 是条件表达式；
- `$"...{value}..."` 是字符串插值。

`var` 并不会把 C# 变成动态类型语言。编译器仍然确定 `taskName` 是 `string`，后续不能把整数赋给它：

```csharp
var taskName = "Windows Runner PoC";
taskName = 42; // 编译错误
```

重新运行：

```bash
dotnet run --project windows-runner/src/WindowsRunner/WindowsRunner.csproj -- "Chrome 启动检查"
```

预期输出：

```text
准备任务：Chrome 启动检查
```

再检查没有参数时的边界情况：

```bash
dotnet run --project windows-runner/src/WindowsRunner/WindowsRunner.csproj
```

预期输出：

```text
准备任务：Windows Runner PoC
```

### Checkpoint

现在程序既能处理调用者传入的任务名称，也能在没有参数时使用明确的默认值。这个完整小程序已经足以练习编辑、编译和运行，不需要为了“像正式项目”而提前抽取类或配置层。

## 读懂 `WindowsRunner.csproj`

项目能够运行，但我们还没有解释 SDK 如何知道它是一个 .NET 8 可执行程序。答案在 `WindowsRunner.csproj`。

控制台模板生成的文件大致如下，具体属性会随 SDK 模板版本略有不同：

```xml
<Project Sdk="Microsoft.NET.Sdk">

  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <ImplicitUsings>enable</ImplicitUsings>
    <Nullable>enable</Nullable>
  </PropertyGroup>

</Project>
```

关键字段的含义是：

| 字段 | 含义 |
| --- | --- |
| `Sdk="Microsoft.NET.Sdk"` | 使用标准 .NET SDK 构建规则 |
| `OutputType=Exe` | 生成可执行应用，而不是类库 |
| `TargetFramework=net8.0` | 面向 .NET 8 API 和运行时契约 |
| `ImplicitUsings=enable` | 自动引入控制台项目常用命名空间 |
| `Nullable=enable` | 开启可空引用类型分析 |

`csproj` 使用 MSBuild 项目格式。现阶段不需要系统学习 MSBuild，只要知道：**项目目标和依赖应由项目文件声明，而不是靠某台电脑上“碰巧安装过什么”来维持。**

## NuGet：依赖如何进入项目

未来 Runner 会使用 Playwright for .NET 控制浏览器。最直接的依赖声明方式是：

```bash
dotnet add windows-runner/src/WindowsRunner/WindowsRunner.csproj package Microsoft.Playwright
```

这条命令完成两件事：

1. 从 NuGet 包源解析 `Microsoft.Playwright`；
2. 把 `PackageReference` 写进 `WindowsRunner.csproj`。

项目文件会增加类似内容：

```xml
<ItemGroup>
  <PackageReference Include="Microsoft.Playwright" Version="实际解析出的版本" />
</ItemGroup>
```

查看当前项目的直接依赖：

```bash
dotnet list windows-runner/src/WindowsRunner/WindowsRunner.csproj package
```

然后执行一次明确的编译检查：

```bash
dotnet build windows-runner/src/WindowsRunner/WindowsRunner.csproj
```

成功时会看到类似：

```text
Build succeeded.
    0 Warning(s)
    0 Error(s)
```

需要注意，**添加 Playwright 包不等于已经完成浏览器 PoC**。它只能证明项目可以解析并编译这个依赖，不能证明：

- 系统 Chrome 能启动；
- 国网所需扩展能工作；
- UKey 驱动能被浏览器访问；
- Windows 原生 PIN 窗口能正常交互；
- 登录后的同一会话能继续被程序读取。

这些结论必须留给真实 Windows 环境。

## `restore`、`build` 和 `run` 到底有什么区别

三个命令经常连续出现，但职责不同：

```text
dotnet restore
    解析 csproj 中的 NuGet 依赖
            |
            v
dotnet build
    restore（默认需要时执行）+ 编译项目
            |
            v
dotnet run
    build（默认需要时执行）+ 启动应用
```

日常开发中，`dotnet build` 和 `dotnet run` 通常会按需还原依赖，所以不必每次手工运行 `dotnet restore`。CI 或故障排查时，把步骤拆开会更容易确定失败发生在哪个阶段。

构建后通常会看到两个目录：

```text
WindowsRunner/
├── bin/
└── obj/
```

- `obj/` 保存依赖解析结果和中间构建文件；
- `bin/` 保存构建输出；
- 两者都属于可再生成内容，通常不提交 Git。

检查仓库现有 `.gitignore` 是否已经包含对应规则；如果没有，再补上：

```gitignore
**/bin/
**/obj/
```

不要在规则已经存在时重复添加，也不要为了清理构建目录顺手改动无关忽略项。

## C# 语法先掌握哪些

为了完成第一版 Runner，不需要先学完整一门语言。优先掌握下面几项即可。以下片段用于预习后续代码，并不是对当前 `Program.cs` 的继续修改。

### 基本类型和类型推断

```csharp
string taskName = "Chrome 启动检查";
int retryCount = 2;
bool isVisible = true;

var profileName = "WindowsRunnerPoc"; // 推断为 string
```

### 条件和代码块

```csharp
if (isVisible)
{
    Console.WriteLine("浏览器将以可见模式启动");
}
else
{
    Console.WriteLine("浏览器不可见");
}
```

### 方法

```csharp
static string DescribeTask(string taskName)
{
    return $"准备任务：{taskName}";
}
```

方法签名明确说明输入是 `string`，输出也是 `string`。这使编译器和调用者都能提前知道契约。

### `async` / `await`

浏览器导航和等待页面元素都需要等待外部事件，Playwright API 因此大量使用异步方法：

```csharp
await page.GotoAsync("https://example.com");
```

`await` 表示异步等待这个操作完成，再继续执行当前流程。它不是“创建一个新线程”的同义词，也不是让所有代码自动并行。第一次写 Runner 时，只需要沿着 Playwright 的异步 API 使用 `await`，不必先设计复杂并发模型。

### `using` 和资源释放

浏览器、文件流和网络连接都占用外部资源。C# 可以使用 `using` 或 `await using` 确保离开作用域时释放资源：

```csharp
using var stream = File.OpenRead("settings.json");
```

后续 Runner 要特别关注浏览器上下文的关闭，但在第一个 `Hello, World!` 项目里还没有资源需要提前封装。

## Linux 开发和 Windows 验收不是一回事

C# 和 .NET 8 是跨平台的。因此，在 WSL 中可以完成：

- 编辑 C#；
- 创建项目；
- 还原 NuGet 依赖；
- 编译；
- 运行普通控制台逻辑；
- 编写不依赖 Windows 的测试。

但 UKey 场景依赖的是一整套本机环境：

```text
Windows 原生 .NET 进程
        |
        v
Windows 系统 Chrome
        |
        v
浏览器扩展 / 证书驱动 / UKey
        |
        v
用户完成 PIN 或其他敏感确认
```

WSL 中的 Linux `dotnet` 进程不等于 Windows 原生进程。它不能因为运行在一台 Windows 电脑里的 WSL2，就自动获得对 Windows UKey 驱动和原生证书窗口的等价访问能力。

因此推荐工作流是：

```text
Linux / WSL
  编写代码、build、单元测试
           |
           v
Git 分支同步代码
           |
           v
原生 Windows
  build、run、系统 Chrome 和 UKey 实机验收
```

Windows 端需要单独安装原生 SDK。在 Windows PowerShell 中可以使用：

```powershell
winget install Microsoft.DotNet.SDK.8
dotnet --info
```

WSL 中安装过 SDK，不代表 Windows PowerShell 已经能找到 `dotnet`；反过来也一样。这是两个独立环境。

## 常见误区

### 误区一：`.NET` 就是 C#

C# 是语言，.NET 是平台。.NET 也支持 F# 和 Visual Basic 等语言。

### 误区二：安装 Runtime 就能创建项目

Runtime 主要用于运行应用。`dotnet new`、编译和开发需要 SDK。

### 误区三：`var` 等于 Python 的动态类型

`var` 只是让编译器推断静态类型。变量的类型在编译时仍然确定。

### 误区四：项目能在 Linux 编译，UKey 就一定能工作

编译成功只证明代码和依赖在当前目标下成立。UKey 兼容性涉及真实 Windows 驱动、Chrome、扩展、权限和原生窗口，必须实机验证。

### 误区五：一开始就需要解决方案文件和多层架构

一个项目时，直接使用 `WindowsRunner.csproj` 足够了。等多个项目真的出现，例如生产项目和测试项目需要统一管理时，再创建 `.sln` 或 `.slnx`。不要为尚未出现的复杂度提前加层。

### 误区六：添加 Playwright 包就应该立刻写完整自动化流程

首轮先验证依赖能够还原和构建。下一篇或下一阶段再只增加一个行为：启动可见的系统 Chrome。登录检测、CA 页面和状态机应在后续压力真正出现时逐步加入。

## 完整练习清单

按下面顺序执行，可以验证本文的学习闭环：

```bash
# 1. 确认 SDK
dotnet --version
dotnet --info

# 2. 创建项目
dotnet new console --framework net8.0 --output windows-runner/src/WindowsRunner

# 3. 运行默认程序
dotnet run --project windows-runner/src/WindowsRunner/WindowsRunner.csproj

# 4. 修改 Program.cs 后传入任务名称
dotnet run --project windows-runner/src/WindowsRunner/WindowsRunner.csproj -- "Chrome 启动检查"

# 5. 添加未来需要的浏览器依赖
dotnet add windows-runner/src/WindowsRunner/WindowsRunner.csproj package Microsoft.Playwright

# 6. 查看依赖并构建
dotnet list windows-runner/src/WindowsRunner/WindowsRunner.csproj package
dotnet build windows-runner/src/WindowsRunner/WindowsRunner.csproj
```

完成后检查这些结果：

- `dotnet --info` 显示 .NET 8 SDK；
- `Program.cs` 能读取命令行参数；
- 无参数时能使用默认任务名称；
- `WindowsRunner.csproj` 面向 `net8.0`；
- 项目文件包含 `Microsoft.Playwright` 的 `PackageReference`；
- `dotnet build` 没有错误；
- Git 不包含 `bin/` 和 `obj/`；
- 你没有把 Linux 构建成功误写成 UKey 已验证。

## 下一小步

现在的程序只会输出任务名称，这是有意保留的边界。下一步只增加一个行为：

> 使用 Playwright for .NET 启动用户可见的系统 Chrome，并使用独立的持久化 profile。

这个下一步需要回答新的具体问题：如何选择系统 Chrome、如何保证 `Headless = false`、profile 放在哪里、关闭程序时怎样释放浏览器资源。只有浏览器启动检查通过后，才应该加入“等待用户登录”和“读取 CA 绑定结果”。

## 参考与延伸阅读

- [.NET 简介](https://learn.microsoft.com/zh-cn/dotnet/core/introduction)
- [C# 语言概览](https://learn.microsoft.com/zh-cn/dotnet/csharp/tour-of-csharp/overview)
- [.NET CLI 概述](https://learn.microsoft.com/zh-cn/dotnet/core/tools/)
- [在 Ubuntu 上安装 .NET](https://learn.microsoft.com/zh-cn/dotnet/core/install/linux-ubuntu-install)
- [什么是 NuGet](https://learn.microsoft.com/zh-cn/nuget/what-is-nuget)
- [Playwright for .NET 入门](https://playwright.dev/dotnet/docs/intro)

## 小结

理解 C#/.NET 的关键不是先背语法，而是分清层次：

```text
C# 描述程序逻辑
.NET SDK 创建并构建项目
csproj 声明目标和依赖
NuGet 提供第三方包
.NET Runtime / CLR 执行构建结果
```

在这个基础上，Windows Runner 就不再是一组陌生命令，而是一个普通的 .NET 8 控制台项目。它可以在 Linux/WSL 中开发和验证跨平台逻辑，但系统 Chrome、证书驱动和 UKey 必须回到原生 Windows 环境完成最后的兼容性证明。
