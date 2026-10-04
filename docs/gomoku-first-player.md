# Gomoku: first-player win and value loss

[繁體中文](gomoku-first-player.zh-TW.md)

## Answer

**Yes. Unrestricted 15×15 Gomoku is a proven win for Black from the starting board, with correct play.** This matches this repo's main rules: Black starts, one stone per turn, five or more in a row wins, with no forbidden moves or swap rule. [1, sections 2 and 7]

This does not mean that Black wins after every opening or later mistake. It means that Black has a strategy from the starting position that wins against every White reply.

## Primary evidence

Allis, van den Herik, and Huntjens reported the result in the 1993 AAAI paper *Go-Moku Solved by New Search Techniques*. [1]

- Section 2 defines the 15×15 board and the unrestricted rules. Overlines, meaning six or more in a row, also win for either player.
- Section 7 reports solving both the unrestricted game and variant B, where only exactly five wins. It states: “Both variants have been solved.” Winning trees for Black were stored and checked for consistency and completeness.
- Sections 5–8 describe Victoria's combination of threat-space search and proof-number search. Bounded TSS alone can miss winning lines.
- Section 1 gives a concrete warning: an earlier champion program, playing Black, reached a position proved lost for Black. A won starting game does not excuse later bad moves.

Allis's doctoral thesis *Searching for Solutions in Games and Artificial Intelligence* was published in 1994. The university's official record confirms the work and year. [2][3]

| Rules | Scope of the cited result |
| --- | --- |
| Unrestricted 15×15, five or more wins | Black has a winning starting strategy |
| 15×15, exactly five wins for either color | Also solved for Black in this paper |
| Renju, with Black-only forbidden patterns | Do not transfer the result |
| Swap/Swap2 opening rules | Do not transfer the result |
| Arbitrary later positions or random starts | The starting-position proof does not assign all their values |

No claim about a current solution of other variants is made here.

## What this means for training

Black's strong results are not, by themselves, evidence of a bug or overfitting. If most self-play games end with Black winning, the model can fit those results with very small value loss. The first-player result makes this less surprising.

But two different values must not be mixed:

- **Perfect-play value:** the result when both players make the best possible decisions.
- **Played-game value target:** the actual terminal result produced by the players and move choices used in that game.

This repo uses the second as its training label, from the player-to-move view: +1 for a win, 0 for a draw, and −1 for a loss. Labels come from real terminal games, not from a starting-position theorem or an unfinished TSS proof. See [`engine/selfplay.py`](../engine/selfplay.py) and [`engine/training.py`](../engine/training.py).

Actual players can make mistakes. Self-play and evaluation also use different starts and move choices. Thus a training/evaluation value-error gap can reflect a change in the games being played, not just memorization. Loss alone cannot separate those causes.

A winning move already available in a later position is a stronger check: its player can win immediately, regardless of the other player's skill. A negative value prediction there is wrong even if the empty board is a Black win.

Our earlier CPU probe of the 8,000-game model found near-zero last-minibatch training value loss but about 0.493 value MSE on 300 saved evaluation positions. That is a warning about transfer and value quality, not proof of classic overfitting. The Black-win theorem does not remove the need for position-level checks.

## Limits of this engine

The engine has its own bounded C++ TSS and independent proof checking. It does not ship Victoria's winning database or a complete proof-number solver. A published winning strategy does not mean this bot follows it.

Keep the existing rules and real terminal labels. Check value error on separate games and on known winning/losing positions before changing the model. No training setting was changed for this research.

## Sources

1. L. V. Allis, H. J. van den Herik, M. P. H. Huntjens (1993), [*Go-Moku Solved by New Search Techniques*](https://cdn.aaai.org/Symposia/Fall/1993/FS-93-02/FS93-02-001.pdf), AAAI Technical Report FS-93-02. See sections 1, 2, and 5–8.
2. Maastricht University, [official thesis record](https://cris.maastrichtuniversity.nl/en/publications/searching-for-solutions-in-games-and-artificial-intelligence/), DOI [10.26481/dis.19940923la](https://doi.org/10.26481/dis.19940923la).
3. L. Victor Allis (1994), [*Searching for Solutions in Games and Artificial Intelligence*](https://project.dke.maastrichtuniversity.nl/games/files/phd/SearchingForSolutions.pdf).
