import type { Plugin } from "@opencode-ai/plugin"

const PROTECTED_DIRS = ["pina", "tests", "docs"]
const EDIT_TOOLS = ["edit", "write", "patch"]

function toPosix(path: string) {
  return path.replace(/\\/g, "/")
}

function relativeToRoot(path: string, root: string) {
  const target = toPosix(path)
  const base = toPosix(root).replace(/\/+$/, "")

  if (target === base) return ""
  if (target.startsWith(`${base}/`)) return target.slice(base.length + 1)
  return target.replace(/^\.\//, "")
}

function isProtected(path: string, root: string) {
  const relative = relativeToRoot(path, root)
  return PROTECTED_DIRS.some(
    (dir) => relative === dir || relative.startsWith(`${dir}/`),
  )
}

function collectStrings(value: unknown, result: string[] = []) {
  if (typeof value === "string") {
    result.push(value)
  } else if (Array.isArray(value)) {
    for (const item of value) collectStrings(item, result)
  } else if (value && typeof value === "object") {
    for (const item of Object.values(value)) collectStrings(item, result)
  }
  return result
}

function pathsFromToolArgs(args: unknown) {
  if (!args || typeof args !== "object") return []

  const paths: string[] = []
  const record = args as Record<string, unknown>

  for (const key of ["filePath", "filepath", "path", "file"]) {
    const value = record[key]
    if (typeof value === "string") paths.push(value)
  }

  if (typeof record.patchText === "string") {
    for (const line of record.patchText.split(/\r?\n/)) {
      const match = line.match(/^\*\*\* (?:Add|Update|Delete) File: (.+)$/)
      if (match) paths.push(match[1].trim())
    }
  }

  return paths
}

function pathsFromPermission(input: {
  pattern?: string | string[]
  metadata?: Record<string, unknown>
  title?: string
}) {
  const paths = collectStrings(input.pattern)
  const metadata = input.metadata ?? {}

  for (const key of ["filePath", "filepath", "path", "file", "files"]) {
    if (key in metadata) collectStrings(metadata[key], paths)
  }

  if (paths.length === 0 && typeof input.title === "string") {
    paths.push(input.title)
  }

  return paths
}

export default (async ({ client, directory, worktree }) => {
  const root = worktree || directory

  async function notify(path: string) {
    try {
      await client.tui.showToast({
        body: {
          title: "Source change requires approval",
          message: `Approve the permission prompt before modifying ${path}`,
          variant: "warning",
          duration: 5000,
        },
      })
    } catch {}
  }

  return {
    "permission.ask": async (input, output) => {
      if (input.type !== "edit") return

      const paths = pathsFromPermission(input)
      if (paths.length === 0) return

      const sourcePath = paths.find((path) => isProtected(path, root))
      if (sourcePath) {
        if (output.status !== "deny") output.status = "ask"
        await notify(sourcePath)
      }
    },

    "tool.execute.before": async (input, output) => {
      if (!EDIT_TOOLS.includes(input.tool)) return

      const paths = pathsFromToolArgs(output.args)
      const sourcePath = paths.find((path) => isProtected(path, root))
      if (sourcePath) await notify(sourcePath)
    },
  }
}) satisfies Plugin
