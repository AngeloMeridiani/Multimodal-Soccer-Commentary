import json

events = json.load(open('src/features/events/possesso_palla_e_parata_enriched.json'))
print("Current enriched events:")
for e in events:
    print(f"t={e['t']:.2f} | type={e.get('type')} | player={e.get('player')} | team={e.get('player_team')}")
