# EV Calculation Logic

## Purpose
This document defines the rules for calculating the Expected Value (EV) of a bet on either the Home or Away team across different Asian Handicap spread values. These formulas are used to evaluate whether a bet has positive or negative expected return based on win probabilities and decimal odds.

## Key Variables
- **'Home Win Probability'** – Probability that the Home team wins the match outright
- **'Away Win Probability'** – Probability that the Away team wins the match outright
- **'Draw Probability'** – Probability that the match ends in a draw
- **'Away -1 Probability'** – Probability that the Away team wins by 2 or more goals (Away team -1 handicap covered)
- **'Home -1 Probability'** – Probability that the Home team wins by 2 or more goals (Home team -1 handicap covered)
- **'Home Odds'** – Decimal odds offered on the Home team bet
- **'Away Odds'** – Decimal odds offered on the Away team bet

> Note: All probabilities should sum to 1 (i.e., Home Win + Draw + Away Win = 1).  
> **'Away -1 Probability'** is a subset of **'Away Win Probability'** (Away wins by 2+).  
> **'Home -1 Probability'** is a subset of **'Home Win Probability'** (Home wins by 2+).

## EV Formula Structure
EV is calculated as:  
**EV = (Probability of winning × Net profit per unit) + (Probability of losing × -1)**

A positive EV indicates a value bet. A negative EV indicates an unfavorable bet.

## Spread Logic
- **Positive spread (e.g. +0.5, +1)** – The Home team receives a handicap advantage. Away team must win by more goals to cover.
- **Negative spread (e.g. -0.5, -1)** – The Away team receives a handicap advantage. Home team must win by more goals to cover.
- **Quarter spreads (±0.25, ±0.75, ±1.25)** – The stake is split into two halves: one on the whole number and one on the half number. This results in partial wins/losses (×0.5 factor) when the result lands on the boundary score.

## Spread = 0
EV Home = 'Home Win Probability'*('Home Odds'-1)+ 'Away Win Probability' *(-1)
EV Away = 'Away Win Probability'*('Away Odds'-1)+ 'Home Win Probability' *(-1)

# Spread = 0.25
EV Home = 'Home Win Probability'*('Home Odds'-1)+ 'Draw Probability'*0.5*('Home Odds'-1)+ 'Away Win Probability' *(-1)
EV Away = 'Away Win Probability'*('Away Odds'-1)+ 'Draw Probability'*-0.5+ 'Home Win Probability' *(-1)

# Spread = 0.5
EV Home = ('Home Win Probability' + 'Draw Probability') *('Home Odds'-1) + 'Away Win Probability' *(-1)
EV Away = 'Away Win Probability'*('Away Odds'-1)+ ('Draw Probability' + 'Home Win Probability') *(-1)

# Spread = 0.75
EV Home = ('Home Win Probability' + 'Draw Probability')*('Home Odds'-1) + ('Away Win Probability' - 'Away -1 Probability') *(-0.5) + ('Away -1 Probability')*(-1)
EV Away = ('Away Win Probability' - 'Away -1 Probability') * ('Away Odds'-1)* 0.5 + ('Away -1 Probability')*('Away Odds'-1)+('Home Win Probability' + 'Draw Probability') *(-1)

# Spread = 1
EV Home = ('Home Win Probability' + 'Draw Probability')*('Home Odds'-1) + ('Away -1 Probability')*(-1)
EV Away = ('Away -1 Probability')*('Away Odds'-1) + ('Home Win Probability' + 'Draw Probability')*(-1)

# Spread = 1.25
EV Home = ('Home Win Probability' + 'Draw Probability')*('Home Odds'-1) + ('Away -1 Probability')*(-1)+('Away Win Probability' - 'Away -1 Probability') *(0.5)*('Home Odds'-1)
EV Away = ('Away -1 Probability')* ('Away Odds'-1) + ('Away Win Probability' - 'Away -1 Probability') *(-0.5) + ('Home Win Probability' + 'Draw Probability')*(-1)

# Spread = 1.5
EV Home = (1- ('Away -1 Probability'))*('Home Odds' -1)+('Away -1 Probability')* (-1)
EV Away = ('Away -1 Probability')* ('Away Odds'-1) + (1- ('Away -1 Probability'))*-1

# Spread = -0.25
EV Home = 'Home Win Probability'*('Home Odds'-1) + 'Draw Probability'*(-0.5) + 'Away Win Probability'*(-1)
EV Away = 'Away Win Probability'*('Away Odds'-1) + 'Draw Probability'*(0.5) + 'Home Win Probability'*(-1)

# Spread = -0.5
EV Home = 'Home Win Probability'*('Home Odds'-1) + ('Draw Probability' + 'Away Win Probability')*(-1)
EV Away = ('Away Win Probability' + 'Draw Probability')*('Away Odds'-1) + 'Home Win Probability'*(-1)

# Spread = -0.75
EV Home = ('Home -1 Probability')*('Home Odds'-1) + ('Home Win Probability' - 'Home -1 Probability')*('Home Odds'-1)*0.5 + ('Away Win Probability' + 'Draw Probability')*(-1)
EV Away = ('Away Win Probability' + 'Draw Probability')*('Away Odds'-1) + ('Home Win Probability' - 'Home -1 Probability')*(-0.5) + ('Home -1 Probability')*(-1)

# Spread = -1
EV Home = ('Home -1 Probability')*('Home Odds'-1) + ('Away Win Probability' + 'Draw Probability')*(-1)
EV Away = ('Away Win Probability' + 'Draw Probability')*('Away Odds'-1) + ('Home -1 Probability')*(-1)

# Spread = -1.25
EV Home = ('Home -1 Probability')*('Home Odds'-1) + ('Home Win Probability' - 'Home -1 Probability')*(-0.5) + ('Away Win Probability' + 'Draw Probability')*(-1)
EV Away = ('Away Win Probability' + 'Draw Probability')*('Away Odds'-1) + ('Home Win Probability' - 'Home -1 Probability')*(0.5) + ('Home -1 Probability')*(-1)

# Spread = -1.5
EV Home = ('Home -1 Probability')*('Home Odds'-1) + (1 - 'Home -1 Probability')*(-1)
EV Away = (1 - 'Home -1 Probability')*('Away Odds'-1) + ('Home -1 Probability')*(-1)