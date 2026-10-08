import { useCallback, useEffect, useState } from 'react'
import { fetchDashboard } from './api'
import type { DashboardResponse } from './types'
import TopBar from './components/TopBar'
import { ROUTES, type Route } from './routes'
import PlayerDrawer from './components/PlayerDrawer'
import Projections from './pages/Projections'
import BuyLists from './pages/BuyLists'
import PlayerStatsPage from './pages/PlayerStats'
import TrackRecord from './pages/TrackRecord'
import SilverToGold from './pages/SilverToGold'
import Profit from './pages/Profit'

function routeFromHash(): Route {
  const r = location.hash.replace(/^#\/?/, '').split('?')[0]
  return (ROUTES.find(x => x.id === r)?.id ?? 'projections') as Route
}

/** ?player=<card uuid> in the hash opens that card, so a forecast can be shared as a link. */
function playerFromHash(): string | null {
  return new URLSearchParams(location.hash.split('?')[1] ?? '').get('player')
}

function setPlayerInHash(uuid: string | null) {
  const [path] = location.hash.split('?')
  const next = `${path || '#/'}${uuid ? `?player=${uuid}` : ''}`
  if (next !== location.hash) history.replaceState(null, '', next)
}

export default function App() {
  const [data, setData] = useState<DashboardResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [route, setRoute] = useState<Route>(routeFromHash)
  const [selected, setSelectedState] = useState<string | null>(playerFromHash)
  const setSelected = useCallback((uuid: string | null) => { setSelectedState(uuid); setPlayerInHash(uuid) }, [])

  useEffect(() => {
    fetchDashboard().then(setData).catch(err => setError(err.message))
  }, [])

  useEffect(() => {
    const onHash = () => { setRoute(routeFromHash()); setSelectedState(playerFromHash()); window.scrollTo(0, 0) }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  useEffect(() => {
    const label = ROUTES.find(r => r.id === route)?.label
    document.title = `${label} · Roster Forecast`
  }, [route])

  const closeDrawer = useCallback(() => setSelected(null), [setSelected])

  const body = error ? (
    <div className="state">
      <strong className="strong">Forecasts didn't load</strong>
      <span>{error}. Check that the API is running, then reload.</span>
      <button className="btn" onClick={() => location.reload()}>Reload</button>
    </div>
  ) : !data ? (
    <div className="state" aria-busy="true">Loading forecasts&hellip;</div>
  ) : route === 'buy-lists' ? (
    <BuyLists data={data} onSelect={setSelected} />
  ) : route === 'silver-gold' ? (
    <SilverToGold data={data} onSelect={setSelected} />
  ) : route === 'profit' ? (
    <Profit data={data} onSelect={setSelected} />
  ) : route === 'stats' ? (
    <PlayerStatsPage data={data} onSelect={setSelected} selected={selected} />
  ) : route === 'track-record' ? (
    <TrackRecord data={data} onSelect={setSelected} />
  ) : (
    <Projections data={data} onSelect={setSelected} selected={selected} />
  )

  const selectedPrediction = data?.predictions.find(p => p.card_uuid === selected)

  return (
    <>
      <a className="visually-hidden" href="#main">Skip to content</a>
      <TopBar route={route} status={data?.update_status ?? null} />
      <main id="main" className="page">{body}</main>
      <footer className="footer">
        <span>Ratings and card art from The Show's public API. Stats from the MLB Stats API.</span>
        <span>Fan project, not affiliated with San Diego Studio or MLB. Forecasts are estimates, not guarantees.</span>
      </footer>
      {selectedPrediction && <PlayerDrawer prediction={selectedPrediction} model={data?.model} onClose={closeDrawer} />}
    </>
  )
}
