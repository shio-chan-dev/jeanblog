---
title: "CSR vs SSR：取舍、性能指标与落地路径"
date: 2026-01-24T11:03:05+08:00
draft: false
description: "从 TTFB/TTI/SEO 等指标出发，对比 CSR 与 SSR 的优缺点，并给出决策指南与可运行示例。"
tags: ["前端", "SSR", "CSR", "性能", "架构"]
categories: ["前端架构"]
keywords: ["CSR", "SSR", "TTFB", "TTI", "Hydration"]
---

## 副标题 / 摘要

CSR 与 SSR 的选择不是二选一，而是围绕性能、SEO、复杂度的权衡。本文给出可操作的决策路径与示例。

## 目标读者

- 负责前端架构选型的工程师
- 需要改善首屏体验与 SEO 的团队
- 希望理解 TTFB/TTI 的开发者

## 背景 / 动机

CSR（客户端渲染）强调前端灵活与交互性，SSR（服务端渲染）强调首屏体验与 SEO。  
很多项目因为选型不当，出现首屏慢、SEO 差或部署复杂度过高的问题。

## 核心概念

- **TTFB**：首字节时间，越小越好
- **TTI**：可交互时间
- **Hydration**：SSR 之后在客户端接管交互
- **SEO**：搜索引擎对 HTML 内容的可见性

## 实践指南 / 步骤

1. **先看内容属性**：是否依赖 SEO、是否内容密集  
2. **评估交互复杂度**：高度交互通常偏向 CSR 或 SSR+Hydration  
3. **关注性能指标**：TTFB、FCP、TTI、CLS  
4. **考虑部署成本**：SSR 需要服务器渲染能力  
5. **混合策略**：关键页 SSR，其余 CSR 或 SSG

## 可运行示例

下面用一个最小 Python 服务演示 SSR 和 CSR 的差异：

```python
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import time

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/ssr":
            html = f"<h1>SSR time: {time.time()}</h1>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html.encode())
        elif self.path == "/csr":
            html = """
            <div id='root'>Loading...</div>
            <script>
              fetch('/api/time').then(r => r.json()).then(d => {
                document.getElementById('root').innerText = 'CSR time: ' + d.time;
              });
            </script>
            """
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html.encode())
        elif self.path == "/api/time":
            body = json.dumps({"time": time.time()}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
```

启动后访问：
- `http://127.0.0.1:8000/ssr`
- `http://127.0.0.1:8000/csr`

## 解释与原理

SSR 在服务端生成 HTML，TTFB 通常更低，SEO 更友好，但服务器负载更高。  
CSR 把渲染推到客户端，首屏可能慢，但交互与开发体验更灵活。  
现代框架通常提供混合模式（SSR + CSR + SSG）。

## 常见问题与注意事项

1. **SSR 一定比 CSR 快吗？**  
   不一定，慢在服务端渲染或缓存失效时可能更慢。

2. **CSR 对 SEO 完全不友好吗？**  
   取决于搜索引擎的渲染能力，但风险更高。

3. **Hydration 的成本高吗？**  
   视页面复杂度而定，重交互页面可能需要更多优化。

## 最佳实践与建议

- 核心落地页/营销页用 SSR 或 SSG
- 高交互后台/应用页用 CSR
- 用缓存与边缘渲染优化 SSR 成本

## 小结 / 结论

CSR 与 SSR 是围绕体验、成本、复杂度的权衡。  
合理的做法是混合使用，根据页面价值与访问场景决定策略。

## 参考与延伸阅读

- Next.js / Nuxt / Remix 官方文档
- Web Vitals 指标说明
- Edge Rendering / SSG 相关资料

## 元信息

- **阅读时长**：9~12 分钟  
- **标签**：CSR、SSR、前端架构  
- **SEO 关键词**：CSR, SSR, TTFB, TTI, Hydration  
- **元描述**：对比 CSR 与 SSR 的性能与架构取舍，并给出实践路径。

## 行动号召（CTA）

挑一条核心页面路径，分别测量 CSR 与 SSR 的首屏指标，你会更明确选型答案。
