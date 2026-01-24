---
title: "JavaScript for 循环闭包陷阱：为什么会打印 3"
date: 2026-01-24T15:29:20+08:00
draft: false
description: "解释 for 循环中的闭包问题，并给出正确修复方式。"
tags: ["Web", "JavaScript", "闭包", "陷阱"]
categories: ["Web"]
keywords: ["JavaScript", "闭包", "for 循环", "let"]
---

## 副标题 / 摘要

for 循环里的闭包经常会打印同一个值。本文解释原因，并给出可运行修复方法。

## 目标读者

- 使用 JavaScript 的开发者
- 需要理解闭包的工程师
- 前端与全栈团队

## 背景 / 动机

JavaScript 的函数作用域与闭包容易导致“循环变量捕获”问题。  
理解这个陷阱能避免常见 Bug。

## 核心概念

- **闭包**：函数捕获外部变量
- **作用域**：var 与 let 的区别
- **事件回调**：延迟执行时才读取变量

## 实践指南 / 步骤

1. **用 let 替代 var**
2. **或使用立即执行函数（IIFE）**
3. **把循环变量变成函数参数**
4. **在回调中避免直接引用 var 变量**

## 可运行示例

```html
<button id="button0">0</button>
<button id="button1">1</button>
<button id="button2">2</button>
<script>
function hookupevents() {
  for (let i = 0; i < 3; i++) {
    document.getElementById("button" + i)
      .addEventListener("click", function() {
        alert(i);
      });
  }
}

hookupevents();
</script>
```

## 解释与原理

使用 var 时，循环结束后 i 的值为 3，闭包读取的是同一个变量。  
用 let 会创建块级作用域，每次循环都有独立 i。

## 常见问题与注意事项

1. **为什么 alert 都是 3？**  
   回调执行时读取的是同一个 i 变量。

2. **用 let 就一定安全吗？**  
   在现代 JS 中是最简单可靠的方式。

3. **旧环境怎么办？**  
   用 IIFE 把 i 作为参数传入。

## 最佳实践与建议

- 默认使用 let/const
- 避免 var 捕获循环变量
- 用测试覆盖关键交互

## 小结 / 结论

for 循环闭包问题来自变量作用域。  
用 let 或 IIFE 能彻底避免。

## 参考与延伸阅读

- MDN: let
- JavaScript Closures

## 元信息

- **阅读时长**：5~7 分钟  
- **标签**：JS、闭包  
- **SEO 关键词**：JavaScript 闭包, for 循环  
- **元描述**：解释 JS for 循环闭包陷阱与修复。

## 行动号召（CTA）

把你项目中的 var 改成 let/const，观察是否减少了闭包问题。
