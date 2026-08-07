export type HealthKey = 'asthma_copd' | 'heart' | 'diabetes' | 'pregnant' | 'limited_mobility' | 'works_outdoors' | 'none';

export const HEALTH_OPTIONS: { key: HealthKey; label: string }[] = [
  { key: 'asthma_copd', label: 'Asthma / COPD' },
  { key: 'heart', label: 'Heart condition' },
  { key: 'diabetes', label: 'Diabetes' },
  { key: 'pregnant', label: 'Pregnant' },
  { key: 'limited_mobility', label: 'Limited mobility' },
  { key: 'works_outdoors', label: 'Works outdoors' },
  { key: 'none', label: 'None of these' },
];

const HEALTH_TO_TAG: Record<Exclude<HealthKey, 'none'>, string> = {
  asthma_copd: 'Respiratory',
  heart: 'Heart',
  diabetes: 'Diabetes',
  pregnant: 'Pregnant',
  limited_mobility: 'Limited Mobility',
  works_outdoors: 'Worker',
};

export const autoRiskForAge = (age: number): string[] => {
  const factors: string[] = [];
  if (age > 0 && age < 12) factors.push('Child');
  if (age > 60) factors.push('Elder');
  return factors;
};

export const healthKeysToTags = (health: HealthKey[], age?: number): string[] => {
  const tags = health
    .filter((key) => key !== 'none')
    .map((key) => HEALTH_TO_TAG[key]);

  if (age) tags.push(...autoRiskForAge(age));

  return Array.from(new Set(tags));
};
