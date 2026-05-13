Top level requirements for the action space:
- noise isn't introduced into the learning process for heads that aren't being used per turn - no incorrect association between action & reward can be made.
- actions look the same between each recurrent action. e.g. purchasing the first container should be the same action process as purchasing the second, the third, etc.
- if we use maskable PPO, the PPO model is stateless. We need some mechanism for it to remember that it's taking a sub-turn step, such as pricing a single container within the whole produce turn.  

Sequential compound actions:
Each action head has an additional "no-op" element. This is used & automatically set when we want the agent to learn that it can't do anything on this head for that turn & the other elements, which mean something, won't be selected.



So for a 4 player game, we have:
| Head index | Name | Size | Description |
|---|---|---|---|
| 0 | **Action Type** | 12 | all | One of the 11 action types + no-op |
| 1 | **Opponent** | 5 | Which opponent to target (clockwise from acting player) + no-op |
| 2 | **Color** | 6 | Which container colour + no-op |
| 3 | **Price Slot** | 11 |  Which price slot ($1–$10) to sell from + no-op |
| ~~4~~ | ~~**Purchase**~~ | ~~`n_colors × PRICE_SLOTS + 1`~~ | ~~Single purchase: `color × PRICE_SLOTS + price_slot`, or STOP at index `n_colors × PRICE_SLOTS`~~ | 

(remove purchase, we already have price)

A typical turn will then look something like this:
A single turn will consistute multiple individual actions.
At the beginning of a turn for an agent, the "no-op" for head 0 (buy factory, etc) is masked, and for every other head, everything except the no-op is masked. Note, in the diagrams of the action spaces below, no-op is always index 0 for each head:

action 1:
```
[~~0~~ 1 2 3 4 5 6 7 8 9 10 11]  # action type
[0 ~~1 2 3 4~~]  ## opponent
[0 ~~1 2 3 4 5~~]  ## color
[0 ~~1 2 3 4 5 6 7 8 9 10~~]  ## price slot
```

Let's say action 1 we choose to do the action: Buy factory. 

We now need to choose a colour. Now, the action masks for all heads except color only allow the no-op action. For colour, we allow each colour to be selected:
Action 2:
```
[0 ~~1 2 3 4 5 6 7 8 9 10 11~~]  # action type
[0 ~~1 2 3 4~~]  ## opponent
[~~0~~ 1 2 3 4 5]  ## color
[0 ~~1 2 3 4 5 6 7 8 9 10~~]  ## price slot
```

Let's say we choose the colour at index 2, and we start the game with the colour at index 1. So now we have 2 factories with colours, [1, 2]

That's the end of turn 1. For turn 2, we can choose any action type again:
Action 3:
```
[~~0~~ 1 2 3 4 5 6 7 8 9 10 11]  # action type
[0 ~~1 2 3 4~~]  ## opponent
[0 ~~1 2 3 4 5~~]  ## color
[0 ~~1 2 3 4 5 6 7 8 9 10~~]  ## price slot
```

Let's say we choose the action: produce. 

We can now produce 2 containers - 1 for each factory we have. We need to select each colour we want to produce and then set a price we want to produce at:

Action 4:
```
[0 ~~1 2 3 4 5 6 7 8 9 10 11~~]  # action type
[0 ~~1 2 3 4~~]  ## opponent
[~~0~~ 1 2 ~~3 4 5~~]  ## color
[0 ~~1 2 3 4 5 6 7 8 9 10~~]  ## price slot
```

We choose colour at index 1. Now we need to set a price for this container.

Action 5:
```
[0 ~~1 2 3 4 5 6 7 8 9 10 11~~]  # action type
[0 ~~1 2 3 4~~]  ## opponent
[0 ~~1 2 3 4 5~~]  ## color
[~~0~~ 1 2 3 4 ~~5 6 7 8 9 10~~]  ## price slot
```

And select the second coloured container.

Action 6:
```
[0 ~~1 2 3 4 5 6 7 8 9 10 11~~]  # action type
[0 ~~1 2 3 4~~]  ## opponent
[~~0 1~~ 2 ~~3 4 5~~]  ## color
[0 ~~1 2 3 4 5 6 7 8 9 10~~]  ## price slot
```

We can only choose the colour at index 2 as that is our only other available option. And select the second price:
```
[0 ~~1 2 3 4 5 6 7 8 9 10 11~~]  # action type
[0 ~~1 2 3 4~~]  ## opponent
[0 ~~1 2 3 4 5~~]  ## color
[~~0~~ 1 2 3 4 ~~5 6 7 8 9 10~~]  ## price slot
```

Note that invalid colour and price slots are also masked out. 