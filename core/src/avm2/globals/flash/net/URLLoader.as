package flash.net {
    import flash.events.EventDispatcher;
    import flash.net.URLRequest;

    public class URLLoader extends EventDispatcher {
        [Ruffle(NativeAccessible)]
        public var data:*;

        [Ruffle(NativeAccessible)]
        public var dataFormat:String = "text";

        [Ruffle(NativeAccessible)]
        public var bytesLoaded:uint;

        [Ruffle(NativeAccessible)]
        public var bytesTotal:uint;

        public function URLLoader(request:URLRequest = null) {
            if (request != null) {
                this.load(request);
            }
        }

        public native function load(request:URLRequest):void;

        public function close():void {
            // Cancellation of the in-flight load is not currently tracked
            // by Ruffle's NavigatorBackend; close() therefore can't actually
            // abort it. Treated as a no-op — the load future runs to
            // completion and its events are still dispatched.
        }
    }
}
