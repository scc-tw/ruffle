package flash.net {
    import flash.net.URLRequest;

    public native function navigateToURL(request:URLRequest, window:String = null):void;

    public native function registerClassAlias(name:String, object:Class):void;
    public native function getClassByAlias(name:String):Class;

    public function sendToURL(request:URLRequest):void {
        // Fire-and-forget HTTP send. We don't yet route this through the
        // navigator (it would need a plain "send, ignore response" path),
        // so the request is silently dropped. Most callers use sendToURL
        // for telemetry and don't care whether it lands.
    }
}
