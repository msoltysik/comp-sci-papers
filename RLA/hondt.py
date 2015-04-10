import math
import numpy as np
import scipy
from scipy.stats import binom
import pandas as pd


def dHondt(partyTotals, seats, divisors):
    '''
    allocate <seats> seats to parties according to <partyTotals> votes,
    using D'Hondt proportional allocation with <weights> divisors

    Input:
        partyTotals: list of total votes by party
        seats: total number of seats to allocate
        divisors: divisors for proportional allocation. For D'Hondt, divisors are 1, 2, 3, ...

    Returns:
        partySeats: list of number of seats for each party
        seated: list of tuples--parties with at least one seat, number of votes that party got, and divisor for last seated in the party notSeated: list of tuples--parties with at least one lost seat, number of votes that party got,
        and divisor for the first non-seated in the party pseudoCandidates: matrix of votes for each pseudocandidate
    '''

    pseudoCandidates = np.array(
        [partyTotals, ] * seats, ).T / divisors.astype(float)
    sortedPC = np.sort(np.ravel(pseudoCandidates))
    lastSeated = sortedPC[-seats]
    theSeated = np.where(pseudoCandidates >= lastSeated)
    partySeats = np.bincount(theSeated[0], minlength=len(partyTotals))
    # number of seats for each party
    inx = np.nonzero(partySeats)[0]  # only those with at least one seat
    seated = zip(inx, partyTotals[inx], divisors[partySeats[inx] - 1])
    # parties with at least one seat,
    # number of votes that party got,
    # and divisor for last seated in
    # the party
    theNotSeated = np.where(pseudoCandidates < lastSeated)
    partyNotSeats = np.bincount(theNotSeated[0], minlength=len(partyTotals))
    # number of non-seats for each
    # party
    inx = np.nonzero(partyNotSeats)[0]
    notSeated = zip(inx, partyTotals[inx], divisors[partySeats[inx]])
    # parties with at least one
    # unseated, number of votes # that party got, and divisor
    # for the first non-seated
    # in the party
    if (lastSeated == sortedPC[-(seats + 1)]):
        raise ValueError("Tied contest for the last seat!")
    else:
        return partySeats, seated, notSeated, lastSeated, pseudoCandidates


def uMax(win, lose):
    '''
    finds the upper bound u on the MICRO for the contest
    win and lose are lists of triples: [party, tally(party), divisor]
    the divisor for win is the largest divisor for any seat the party won
    the divisor for lose is the smallest divisor for any seat the party lost
    See Stark and Teague, 2014, equations 4 and 5.

    Input:
        win: list of triples--party, tally(party), divisor
        lose: list of triples--party, tally(party), divisor

    Returns:
        maximum possible relative overstatement for any ballot
    '''
    u = 0.0
    for w in win:
        for ell in lose:
            if w[0] != ell[0]:
                u = max(
                    [u, (float(ell[2]) + float(w[2])) / float(ell[2] * w[1] - w[2] * ell[1])])
    return u


def minSampleSize(ballots, u, gamma=0.95, alpha=0.1):
    '''
    find smallest sample size for risk-limit alpha, using cushion gamma \in (0,1)
    1/alpha = (gamma/(1-1/(ballots*u))+1-gamma)**n

    Input:
        ballots: number of ballots cast in the contest
        u: upper bound on overstatement per ballot
        gamma: hedge against finding a ballot that attains the upper bound.
        Larger values give less protection
        alpha: risk limit
    '''
    return math.ceil(math.log(1.0 / alpha) / math.log(gamma / (1.0 - 1.0 / (ballots * u)) + 1.0 - gamma))


# final 2014 Danish EU Parliamentary election results from
# http://www.dst.dk/valg/Valg1475795/valgopg/valgopgHL.htm
# there were two coalitions: (A,B,F) and (C,V)
# There were 13 seats to allocate.
#
# Official results by party
#
A = 435245
B = 148949
C = 208262
F = 249305
I = 65480
N = 183724
O = 605889
V = 379840
Ballots = 2332217  # includes invalid and blank ballots
nSeats = 13  # seats to allocate
#
# allocate seats to coalitions
#
coalitionTotals = np.array([A + B + F, C + V, I, N, O])  # for coalitions
coalitionSeats, coalitionSeated, coalitionNotSeated, coalitionLastSeated, coalitionPCs = dHondt(
    coalitionTotals, nSeats, np.arange(1, nSeats + 1))
print 'A+B+F, C+V, I, N, O:', coalitionSeats
#
# allocate seats within coalitions
#
nABFSeats = coalitionSeats[0]
nCVSeats = coalitionSeats[1]
ABFSeats, ABFSeated, ABFNotSeated, ABFLastSeated, ABFPCs = dHondt(
    np.array([A, B, F]), nABFSeats, np.arange(1, nABFSeats + 1))
CVSeats, CVSeated, CVNotSeated, CVLastSeated, CVPCs = dHondt(
    np.array([C, V]), nCVSeats, np.arange(1, nCVSeats + 1))

print 'A, B, F:', ABFSeats, '; C, V:', CVSeats

ASeats = ABFSeats[0]
BSeats = ABFSeats[1]
CSeats = CVSeats[0]
FSeats = ABFSeats[2]
ISeats = coalitionSeats[2]
NSeats = coalitionSeats[3]
OSeats = coalitionSeats[4]
VSeats = CVSeats[1]
allSeats = [ASeats, BSeats, CSeats, FSeats, ISeats, NSeats, OSeats, VSeats]
print '---------------\nSeats to parties A, B, C, F, I, N, O, V: ', allSeats
print 'Seated coalitions, votes, divisor:', coalitionSeated
print 'Non-Seated coalitions, votes, divisor:', coalitionNotSeated

# Set audit parameters
gamma = 0.95  # tuning constant in the Kaplan-Wald method
alpha = 0.001  # risk limit
#
u = uMax(coalitionSeated, coalitionNotSeated)
print Ballots * u
n = math.ceil(math.log(1.0 / alpha) /
              math.log(gamma / (1.0 - 1.0 / (Ballots * u)) + 1.0 - gamma))
print n
