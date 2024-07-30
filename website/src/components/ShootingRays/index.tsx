import styles from "./styles.module.css";

export default function ShootingRays(): JSX.Element {
  return (
    <div className={styles.container}>
      <span className={styles.ray}></span>
      <span className={styles.ray}></span>
      <span className={styles.ray}></span>
    </div>
  );
}
