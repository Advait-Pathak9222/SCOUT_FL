# Cover letter — IEEE Transactions on Cognitive Communications and Networking

**Manuscript:** SCOUT-FL: Cognitive Client Scheduling for ISAC-Enabled Over-the-Air Federated Learning

---

Dear Editor-in-Chief,

Please find enclosed our manuscript, *"SCOUT-FL: Cognitive Client Scheduling for ISAC-Enabled
Over-the-Air Federated Learning,"* which we submit for consideration as a regular paper in the
IEEE Transactions on Cognitive Communications and Networking.

**Problem.** In an integrated sensing and communication (ISAC) network that trains a model by
over-the-air federated learning, the same uplink transmission carries the learning update and
illuminates the targets. The per-round choice of which devices transmit therefore fixes both the
convergence of the model and the conditioning of the sensing estimate, and it must do so subject
to an aggregation error that the shared multiple-access channel actually delivers. Existing
schedulers optimise one of these objectives and treat the other, and usually the radio itself, as
an abstraction.

**Contribution.** We formulate the joint scheduling-and-transmission problem under a physical link
budget, per-device transmit-power budgets, an interference-inclusive noise floor and a round
latency deadline. We prove that channel-inversion power control is jointly optimal in closed form
and separable from the scheduling decision, so the joint problem collapses without loss of
optimality onto the active set, with the physical layer surviving as a hard constraint on the
weakest active link. The resulting selection rule scores candidate sets by a monotone submodular
utility that couples gradient-space coverage with the log-determinant of the per-target Fisher
information matrix, and enforces the aggregation-error budget through a primal–dual price rather
than a hard gate. Greedy selection carries the (1−1/e) guarantee with a curvature refinement, the
dual loop is feasible in time average and no-regret, and a convergence bound converts the
prescribed error budget into an explicit guarantee on the trained model.

**Why this journal.** The scheduler is a cognitive control loop in the classical sense, and the
paper is organised around that loop rather than around a static optimisation. In every round the
system *perceives* the channel gains, the device geometry and the gradient embeddings; *decides*
the active set by greedy maximisation; *acts* by triggering the joint learning-and-sensing
transmission; and *learns* by updating a dual price from the aggregation error that the
transmission actually incurred. That price is the memory of the loop. One consequence we develop
explicitly is that co-channel interference enters the scheduler only through the effective noise
floor, so the loop absorbs a shift in the interference environment without ever estimating it — it
responds to the consequence of the interference on the task rather than to a model of its source.
We regard this as cognition at the scheduling layer, and it places the work squarely within the
scope of TCCN, alongside recent work in this journal on cognition-driven client scheduling, on
cognitive radios that allocate spectrum jointly to communication and sensing, and on
learning-driven ISAC resource allocation, all of which we cite and position against.

**Evaluation.** The method is evaluated against eleven sensing-aware baselines, including the
closest published ISAC-FL scheduler, over a factorial campaign spanning data heterogeneity and
partition, five datasets, channel model, a 35 dB range of transmit power, and target geometry,
with five seeds per configuration. Comparisons are restricted to methods that carry an explicit
sensing objective, since a learning-only scheduler can buy accuracy simply by declining to sense,
and such a comparison would not measure the quality of the trade-off.

**Declarations.** This manuscript is original, has not been published previously, and is not under
consideration by any other journal or conference. All authors have approved the submission and
declare no conflicts of interest. Any data and code required to reproduce the reported results
will be made available on request.

We believe the work will interest the readership of TCCN and thank you for considering it.

Yours sincerely,

*[author names and affiliations]*
