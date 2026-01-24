---
title: "泛型协变与逆变：为什么 List<Cat> 不是 List<Animal>"
date: 2026-01-24T13:08:42+08:00
draft: false
description: "解释泛型的协变/逆变/不变，并用实际例子说明原因。"
tags: ["语言设计", "泛型", "类型系统", "安全"]
categories: ["语言设计"]
keywords: ["协变", "逆变", "泛型", "类型安全"]
---

## 副标题 / 摘要

很多人困惑为什么 List<Cat> 不能当作 List<Animal>。本文用类型安全的角度解释协变与逆变。

## 目标读者

- 学习泛型与类型系统的开发者
- 需要写类型安全 API 的工程师
- 做语言与框架设计的人

## 背景 / 动机

如果泛型随意协变，会引发类型不安全。  
理解协变/逆变能帮助你正确设计接口与集合使用方式。

## 核心概念

- **协变**：子类型关系在泛型中保留
- **逆变**：子类型关系方向相反
- **不变**：泛型类型不随子类型变化

## 实践指南 / 步骤

1. **只读集合可用协变**
2. **只写集合可用逆变**
3. **读写同时存在时保持不变**
4. **用通配符或泛型参数表达意图**

## 可运行示例

```java
import java.util.List;

class Animal {}
class Cat extends Animal {}

public class VarianceDemo {
    public static void main(String[] args) {
        List<Cat> cats = List.of(new Cat());
        List<? extends Animal> animals = cats; // 协变：只读
        Animal a = animals.get(0);
        System.out.println(a.getClass().getSimpleName());
    }
}
```

## 解释与原理

如果允许 List<Cat> 当作 List<Animal>，就可能把 Dog 放进去，破坏类型安全。  
因此多数语言让泛型默认不变，需要明确声明协变/逆变。

## 常见问题与注意事项

1. **协变集合能写入吗？**  
   不能，通常只能读。

2. **逆变集合能读取吗？**  
   读取会丢失具体类型信息。

3. **为什么这么复杂？**  
   为了在灵活性与类型安全之间取得平衡。

## 最佳实践与建议

- API 设计先区分“只读”与“只写”
- 对外暴露尽量使用协变接口
- 对内实现保持不变类型

## 小结 / 结论

泛型协变与逆变是类型安全的代价，也是接口设计的基础。  
理解它能让你写出更安全的泛型 API。

## 参考与延伸阅读

- Java Generics and Collections
- Kotlin Variance

## 元信息

- **阅读时长**：6~8 分钟  
- **标签**：泛型、类型系统  
- **SEO 关键词**：协变, 逆变, 泛型  
- **元描述**：解释泛型协变/逆变与类型安全。

## 行动号召（CTA）

检查你项目中的泛型接口，看看是否能更明确地表达读写意图。
