import clsx from "clsx";
import Link from "@docusaurus/Link";
import { IconDefinition } from "@fortawesome/fontawesome-svg-core";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import styles from "./styles.module.css";

type Props = {
  children: string;
  to: string;
  icon?: IconDefinition;
  variant?: "primary" | "secondary";
};

export default function Button({
  to,
  children,
  icon,
  variant = "primary",
}: Props): JSX.Element {
  return (
    <Link to={to} className={`button button--${variant} button--lg`}>
      {icon ? (
        <FontAwesomeIcon
          icon={icon}
          className={clsx("margin-right--sm", styles.icon)}
        />
      ) : null}
      {children}
    </Link>
  );
}
