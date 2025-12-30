---
title: "How to Use and Configure a TypeScript Environment"
date: 2025-08-28
draft: false
---

# Introduction

For TypeScript files with the `.ts` extension, we cannot run them directly. We need to transpile TypeScript to JavaScript and then run the JavaScript output.

There are two common approaches: upload `.ts` to the server and compile via CI, or transpile locally and upload the `.js` build to production. If you want to run and test locally during development, you can use `ts-node`, but the project still needs a build step for production.
