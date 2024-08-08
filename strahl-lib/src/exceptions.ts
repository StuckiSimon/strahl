export class SignalAlreadyAbortedError extends Error {
  constructor() {
    super(
      "Signal is already aborted. Make sure to check the aborted property before calling any methods.",
    );
  }
}
