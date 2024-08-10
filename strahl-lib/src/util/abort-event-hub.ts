export function buildAbortEventHub() {
  const instanceState = {
    isRunning: true,
    abortNotifiers: new Map<string, () => void>(),
  };

  return {
    isRunning: () => instanceState.isRunning,
    setDestructionNotifier: (id: string, notifier: () => void) => {
      /**
       * Instantly invoke notifier if instance is already aborted.
       *
       * During async operations it is possible that isRunning is updated before the notifier is set.
       * This ensures that the notifier is invoked in such edge cases.
       */
      if (instanceState.isRunning) {
        instanceState.abortNotifiers.set(id, notifier);
      } else {
        notifier();
      }
    },
    triggerAbort: () => {
      instanceState.isRunning = false;
      for (const [, notifier] of instanceState.abortNotifiers) {
        notifier();
      }
      instanceState.abortNotifiers.clear();
    },
  };
}
