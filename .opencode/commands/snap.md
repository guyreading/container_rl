---
description: Grab screenshots and act on them — explain (huh), fix bugs, remix patterns (do), or make infographics
---

You are helping the user visually through their screenshots directory. Execute these steps precisely.

## STEP 1: List screenshots by modification time (newest first)

Run this shell command first:
```bash
ls -t screenshots/ 2>/dev/null
```
If the output is empty, tell the user "No screenshots found in screenshots/" and stop.

## STEP 2: Parse the user's arguments

The full argument string is: `$ARGUMENTS`

Determine COUNT and ACTION as follows:

- If `$ARGUMENTS` is empty → COUNT = 1, ACTION = "huh"
- If the first token of `$ARGUMENTS` is a positive integer (e.g. 1, 3, 24) → COUNT = that number, ACTION = the rest (trimmed). If the rest is empty, ACTION = "huh"
- If the first token is NOT a number → COUNT = 1, ACTION = the entire `$ARGUMENTS` string

## STEP 3: Grab the screenshots

From the file list in Step 1, take the first COUNT files (they are already newest-first). Read each one using the Read tool. These are image files — analyze and describe what you see in each one.

## STEP 4: Perform the action

Resolve ACTION (case-insensitive, first word match) to one of the following and execute:

### "huh" (or if ACTION is empty)
Describe each screenshot clearly. What do you see? Text, UI elements, code, error messages, diagrams? What is the user likely trying to communicate or ask about? Be observant and thorough.

### "fix"
You are looking at screenshots of bugs, error messages, broken UIs, or failing tests. For each screenshot:
1. Identify the specific error or problem shown
2. Find the relevant source code in the project
3. Edit the code to fix the issue
4. Explain what was wrong and what you changed

If the screenshot shows a visual/design problem (overlapping text, layout issues, wrong colors), find the relevant styling/template code and fix it.

### "do" or "do this"
The screenshot shows something smart, elegant, or effective — a design pattern, algorithm, UI layout, data visualization, architecture, or technique. Learn from it. Then:
1. Explain what the screenshot shows and why it's effective
2. Implement the same approach in the user's project in the most goal-oriented way
3. Adapt and remix it — don't copy blindly. Make it fit the project's existing patterns and the user's needs

### "make" or "infographic" or "make infographic"
Synthesize all the grabbed screenshots into a single unified infographic. Use ASCII art, structured tables, diagrams, and markdown formatting. Highlight:
- Key patterns and insights
- Connections between the screenshots
- A clear, scannable visual summary

### Anything else (custom action)
Treat the entire ACTION string as a custom instruction. Read the screenshots and fulfill that instruction to the best of your ability. Use the full content of every grabbed screenshot.

---

Reminder: you have already listed the screenshots and read the image files. You have their full content. Now execute the resolved action and show the user your results.
