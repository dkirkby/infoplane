"""Replication of the information-plane dynamics of Shwartz-Ziv & Tishby (2017).

Phase 1 of the project: train small fully connected networks, estimate the mutual
information I(X;T) / I(T;Y) of every hidden layer at log-spaced epochs, and export
the ensemble of trajectories for the Phase 2 frontend.
"""

__version__ = "0.1.0"
